// spmv.cpp  (modulo distribution kept)
// Changes vs your version:
// 1) NO unordered_map lookup inside SpMV loop (precompute per-nnz source + index)
// 2) Precompute local positions for exchange_x_values() (no local_index each iter)
// 3) Everything else (including owner(i)=i%P distribution) stays the same

#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>

struct Triplet {
    int r;
    int c;
    double v;
};

static MPI_Datatype make_mpi_triplet() {
    MPI_Datatype T;
    int blocklen[3] = {1,1,1};
    MPI_Aint disp[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

    Triplet tmp;
    MPI_Aint base;
    MPI_Get_address(&tmp, &base);
    MPI_Get_address(&tmp.r, &disp[0]);
    MPI_Get_address(&tmp.c, &disp[1]);
    MPI_Get_address(&tmp.v, &disp[2]);
    disp[0] -= base; disp[1] -= base; disp[2] -= base;

    MPI_Type_create_struct(3, blocklen, disp, types, &T);
    MPI_Type_commit(&T);
    return T;
}

// REQUIRED: modulo distribution stays
static inline int owner(int i, int P) { return i % P; }

static int local_rows_count(int nr, int P, int rank) {
    if (rank >= nr) return 0;
    return (nr - 1 - rank) / P + 1;
}

static int local_cols_count(int nc, int P, int rank) {
    if (rank >= nc) return 0;
    return (nc - 1 - rank) / P + 1;
}

static inline int local_index(int g, int P, int rank) {
    return (g - rank) / P;
}

static std::string matrix_stem(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

static bool file_exists(const std::string& p) {
    std::ifstream f(p.c_str());
    return f.good();
}

static int getenv_int(const char* k, int defv) {
    const char* s = std::getenv(k);
    if (!s || !*s) return defv;
    return std::atoi(s);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank=0, P=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [out_dir]\n";
        MPI_Finalize();
        return 1;
    }

    const std::string file = argv[1];
    const std::string out_dir = (argc >= 3) ? argv[2] : "results";

    if (rank == 0) {
        // note: "mkdir -p" is fine on Linux clusters; keep as you had
        std::string cmd = "mkdir -p " + out_dir;
        (void)std::system(cmd.c_str());
    }

    int nr=0, nc=0, nz=0;
    std::vector<Triplet> A;
    std::vector<Triplet> local_coo;

    MPI_Datatype MPI_TRIPLET = make_mpi_triplet();

    // -------- Read matrix on rank 0 --------
    if (rank == 0) {
        FILE* f = std::fopen(file.c_str(), "r");
        if (!f) {
            std::cerr << "ERROR: cannot open " << file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[512];
        do {
            if (!std::fgets(line, sizeof(line), f)) {
                std::cerr << "ERROR: invalid MatrixMarket header\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        } while (line[0] == '%');

        if (std::sscanf(line, "%d %d %d", &nr, &nc, &nz) != 3) {
            std::cerr << "ERROR: cannot read nr nc nz\n";
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        A.reserve((size_t)nz);
        for (int i=0; i<nz; i++) {
            int a,b; double v;
            if (!std::fgets(line, sizeof(line), f)) {
                std::cerr << "ERROR: unexpected EOF while reading triples\n";
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
            if (std::sscanf(line, "%d %d %lf", &a, &b, &v) != 3) {
                std::cerr << "ERROR: bad triple line\n";
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
            A.push_back({a-1, b-1, v});
        }
        std::fclose(f);

        std::sort(A.begin(), A.end(), [](const Triplet& x, const Triplet& y){
            return (x.r==y.r) ? (x.c < y.c) : (x.r < y.r);
        });
    }

    MPI_Bcast(&nr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int local_nrows = local_rows_count(nr, P, rank);

    // -------- Distribute COO (by row owner = r % P) --------
    if (rank == 0) {
        std::vector<std::vector<Triplet>> send_buf(P);
        for (const auto& t : A)
            send_buf[ owner(t.r, P) ].push_back(t);

        for (int p=1; p<P; p++) {
            int count = (int)send_buf[p].size();
            MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            if (count > 0)
                MPI_Send(send_buf[p].data(), count, MPI_TRIPLET, p, 1, MPI_COMM_WORLD);
        }
        local_coo = std::move(send_buf[0]);
    } else {
        int count=0;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_coo.resize((size_t)count);
        if (count > 0)
            MPI_Recv(local_coo.data(), count, MPI_TRIPLET, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    const int local_nnz = (int)local_coo.size();

    // -------- Build CSR --------
    std::vector<int> row_ptr((size_t)local_nrows + 1, 0);
    std::vector<int> col_idx((size_t)local_nnz);
    std::vector<double> val((size_t)local_nnz);

    for (const auto& t : local_coo) {
        int lr = local_index(t.r, P, rank);
        row_ptr[(size_t)lr + 1]++;
    }
    for (int i=0; i<local_nrows; i++) row_ptr[(size_t)i+1] += row_ptr[(size_t)i];

    std::vector<int> cursor = row_ptr;
    for (const auto& t : local_coo) {
        int lr = local_index(t.r, P, rank);
        int pos = cursor[(size_t)lr]++;
        col_idx[(size_t)pos] = t.c;
        val[(size_t)pos]     = t.v;
    }

    // -------- Distributed x --------
    const int local_ncols = local_cols_count(nc, P, rank);
    std::vector<double> x_local((size_t)local_ncols, 1.0);
    std::vector<double> y_local((size_t)local_nrows, 0.0);

    // -------- Precompute remote x indices --------
    std::vector<std::vector<int>> req_idx(P);
    {
        std::vector<std::unordered_set<int>> seen(P);
        for (int k=0; k<local_nnz; k++) {
            int j = col_idx[(size_t)k];
            int p = owner(j, P);
            if (p != rank) seen[p].insert(j);
        }
        for (int p=0; p<P; p++)
            if (p != rank)
                req_idx[p].assign(seen[p].begin(), seen[p].end());
    }

    std::vector<int> send_counts(P,0), recv_counts(P,0);
    for (int p=0; p<P; p++) send_counts[p] = (int)req_idx[p].size();

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> sdispls(P,0), rdispls(P,0);
    int stot=0, rtot=0;
    for (int p=0; p<P; p++) { sdispls[p]=stot; stot+=send_counts[p]; }
    for (int p=0; p<P; p++) { rdispls[p]=rtot; rtot+=recv_counts[p]; }

    std::vector<int> send_idx_flat((size_t)stot), recv_idx_flat((size_t)rtot);
    {
        int off=0;
        for (int p=0; p<P; p++)
            for (int j : req_idx[p])
                send_idx_flat[(size_t)off++] = j;
    }

    MPI_Alltoallv(send_idx_flat.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  recv_idx_flat.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Map global column index j -> position in recv_val_flat (only used in setup now)
    std::unordered_map<int,int> pos_in_recv;
    pos_in_recv.reserve((size_t)stot * 2);
    for (int i=0; i<stot; i++) pos_in_recv[send_idx_flat[(size_t)i]] = i;

    std::vector<double> send_val_flat((size_t)rtot), recv_val_flat((size_t)stot);

    const int NITER  = getenv_int("NITER", 10);
    const int WARMUP = getenv_int("WARMUP", 1);

    // -------- OPT 1: precompute local positions for values we must SEND (recv_idx_flat are columns owned by this rank)
    std::vector<int> recv_local_pos((size_t)rtot);
    for (int t=0; t<rtot; t++) {
        int j = recv_idx_flat[(size_t)t]; // j is owned by this rank
        recv_local_pos[(size_t)t] = local_index(j, P, rank);
    }

    auto exchange_x_values = [&]() {
        for (int t=0; t<rtot; t++) {
            send_val_flat[(size_t)t] = x_local[(size_t)recv_local_pos[(size_t)t]];
        }
        MPI_Alltoallv(send_val_flat.data(), recv_counts.data(), rdispls.data(), MPI_DOUBLE,
                      recv_val_flat.data(), send_counts.data(), sdispls.data(), MPI_DOUBLE,
                      MPI_COMM_WORLD);
    };

    // -------- OPT 2: precompute for each nnz where to read x from (no hash lookup in kernel)
    std::vector<uint8_t> x_is_remote((size_t)local_nnz, 0);
    std::vector<int>     x_idx((size_t)local_nnz, 0);

    for (int k = 0; k < local_nnz; k++) {
        int j = col_idx[(size_t)k];
        int p = owner(j, P);
        if (p == rank) {
            x_is_remote[(size_t)k] = 0;
            x_idx[(size_t)k] = local_index(j, P, rank);   // index in x_local
        } else {
            x_is_remote[(size_t)k] = 1;
            auto it = pos_in_recv.find(j);
            if (it == pos_in_recv.end()) {
                std::cerr << "Rank " << rank << " missing remote x index for j=" << j << "\n";
                MPI_Abort(MPI_COMM_WORLD, 99);
            }
            x_idx[(size_t)k] = it->second;                // index in recv_val_flat
        }
    }

    auto spmv_local = [&]() {
        for (int i=0; i<local_nrows; i++) {
            double sum = 0.0;
            int k0 = row_ptr[(size_t)i];
            int k1 = row_ptr[(size_t)i + 1];
            for (int k=k0; k<k1; k++) {
                double xj = x_is_remote[(size_t)k]
                    ? recv_val_flat[(size_t)x_idx[(size_t)k]]
                    : x_local[(size_t)x_idx[(size_t)k]];
                sum += val[(size_t)k] * xj;
            }
            y_local[(size_t)i] = sum;
        }
    };

    if (WARMUP) {
        exchange_x_values();
        spmv_local();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // per-iteration raw timings (seconds)
    std::vector<double> comm_it((size_t)NITER), comp_it((size_t)NITER), total_it((size_t)NITER);

    for (int it=0; it<NITER; it++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        exchange_x_values();
        double t1 = MPI_Wtime();
        spmv_local();
        double t2 = MPI_Wtime();

        comm_it[(size_t)it]  = (t1 - t0);
        comp_it[(size_t)it]  = (t2 - t1);
        total_it[(size_t)it] = (t2 - t0);
    }

    // gather raw times to rank 0 (no averages)
    std::vector<double> comm_all, comp_all, total_all;
    if (rank == 0) {
        comm_all.resize((size_t)P * (size_t)NITER);
        comp_all.resize((size_t)P * (size_t)NITER);
        total_all.resize((size_t)P * (size_t)NITER);
    }

    MPI_Gather(comm_it.data(),  NITER, MPI_DOUBLE, rank==0 ? comm_all.data()  : nullptr, NITER, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(comp_it.data(),  NITER, MPI_DOUBLE, rank==0 ? comp_all.data()  : nullptr, NITER, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(total_it.data(), NITER, MPI_DOUBLE, rank==0 ? total_all.data() : nullptr, NITER, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // write CSV (rank 0) -> OUTDIR/<nomematrice>_strong.csv
    if (rank == 0) {
        const std::string stem = matrix_stem(file);
        const std::string csv_path = out_dir + "/" + stem + "_strong.csv";

        if (!file_exists(csv_path)) {
            std::ofstream hdr(csv_path.c_str(), std::ios::out);
            hdr << "# matrix=" << stem << "\n";
            hdr << "# rows=" << nr << "\n";
            hdr << "# nnz_total=" << nz << "\n";
            hdr << "# nnz_per_row=" << ((nr > 0) ? ((double)nz / (double)nr) : 0.0) << "\n";
            hdr << "P,iter,rank,comm_us,comp_us,total_us,gflops,nnz\n";
        }

        std::ofstream out(csv_path.c_str(), std::ios::app);
        out << std::setprecision(15);

        for (int r=0; r<P; r++) {
            for (int it=0; it<NITER; it++) {
                size_t idx = (size_t)r * (size_t)NITER + (size_t)it;

                double ct = comm_all[idx];
                double pt = comp_all[idx];
                double tt = total_all[idx];

                double comm_us  = ct * 1e6;
                double comp_us  = pt * 1e6;
                double total_us = tt * 1e6;

                // Keep your original convention (nz global with per-rank tt),
                // to avoid breaking downstream plotting scripts.
                double gflops = (tt > 0.0) ? (2.0 * (double)nz) / (tt * 1e9) : 0.0;

                out << P << "," << it << "," << r << ","
                    << comm_us << "," << comp_us << "," << total_us << ","
                    << gflops << "," << nz << "\n";
            }
        }

        std::cout << "Wrote " << csv_path << " (P=" << P << ", NITER=" << NITER << ")\n";
    }

    MPI_Type_free(&MPI_TRIPLET);
    MPI_Finalize();
    return 0;
}

