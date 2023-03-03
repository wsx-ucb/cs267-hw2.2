#include "common.h"
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>

using std::vector;

void apply_force(particle_t& particle, particle_t& neighbor, bool bidirectional) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
    if (bidirectional) {
        neighbor.ax -= coef * dx;
        neighbor.ay -= coef * dy;
    }
}

void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

class Group {
  public:
    int i, j, m, n;
    int neignbors[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    vector<vector<particle_t>> bins;
    Group() {};
    Group(int id);
    vector<particle_t>& bin(int i, int j) {
        return bins[i + 1 + (j + 1) * (m + 2)];
    }
    // 1 0 7
    // 2   6
    // 3 4 5
    void inner(std::function<void(vector<particle_t>&, int)> f) {
        for (int d : {0, 2, 4, 6}) {
            int bound = m, other = n;
            if (d % 4) std::swap(bound, other);
            int j = d < 4 ? 0 : other - 1;
            for (int i = 0; i < bound; i++) {
                f(d % 4 ? bin(j, i) : bin(i, j), d);
            }
            int i = d % 6 ? bound - 1 : 0;
            f(d % 4 ? bin(j, i) : bin(i, j), d + 1);
        }
    }
    void outer(std::function<void(vector<particle_t>&, int)> f) {
        for (int d : {0, 2, 4, 6}) {
            int bound = m, other = n;
            if (d % 4) std::swap(bound, other);
            int j = d < 4 ? -1 : other;
            for (int i = 0; i < bound; i++) {
                f(d % 4 ? bin(j, i) : bin(i, j), d);
            }
            int i = d % 6 ? bound : -1;
            f(d % 4 ? bin(j, i) : bin(i, j), d + 1);
        }
    }
    bool contains(int x, int y) {
        return 0 <= x && x < m && 0 <= y && y < n;
    }
};

static int bn, gs, gw;
static Group g;
static double comp_time;

void helper(int m, int n, int i, int& start, int& range) {
    int s = m / n;
    int r = m % n;
    start = i * s + std::min(i, r);
    range = s + (i < r ? 1 : 0);
}

Group::Group(int id) {
    if (id >= gs * gw) return;
    helper(bn, gw, id % gw, i, m);
    helper(bn, gs, id / gw, j, n);
    bool left = i > 0;
    bool right = i + m < bn;
    bool up = j > 0;
    bool down = j + n < bn;
    // 1 0 7
    // 2   6
    // 3 4 5
    if (left) {
        neignbors[2] = id - 1;
        if (up) neignbors[1] = id - 1 - gw;
        if (down) neignbors[3] = id - 1 + gw;
    }
    if (right) {
        neignbors[6] = id + 1;
        if (up) neignbors[7] = id + 1 - gw;
        if (down) neignbors[5] = id + 1 + gw;
    }
    if (up) neignbors[0] = id - gw;
    if (down) neignbors[4] = id + gw;
    bins.resize((m + 2) * (n + 2));
}

void transmit_edges() {
    vector<MPI_Request> requests;
    requests.reserve((g.m + g.n + 2) * 2);
    auto send = [&requests](vector<particle_t>& bin, int dir) {
        if (g.neignbors[dir] == -1) return;
        requests.emplace_back(MPI_Request());
        MPI_Isend(bin.data(), bin.size(), PARTICLE, g.neignbors[dir], (dir + 4) % 8, MPI_COMM_WORLD,
                  &requests.back());
    };
    auto receive = [](vector<particle_t>& bin, int dir) {
        if (g.neignbors[dir] == -1) return;
        MPI_Status status;
        MPI_Probe(g.neignbors[dir], dir, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        bin.resize(count);
        MPI_Recv(&bin[0], count, PARTICLE, g.neignbors[dir], dir, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    };
    // Send inner boundary
    g.inner(send);
    // Receive outer boundary
    g.outer(receive);
    // Wait for all sends
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void transmit_edges2() {
    vector<MPI_Request> requests;
    requests.reserve((g.m + g.n + 2) * 2);
    auto send = [&requests](vector<particle_t>& bin, int dir) {
        if (g.neignbors[dir] == -1) return;
        requests.emplace_back(MPI_Request());
        MPI_Isend(bin.data(), bin.size(), PARTICLE, g.neignbors[dir], (dir + 4) % 8, MPI_COMM_WORLD,
                  &requests.back());
    };
    auto receive = [](vector<particle_t>& bin, int dir) {
        if (g.neignbors[dir] == -1) return;
        MPI_Status status;
        MPI_Probe(g.neignbors[dir], dir, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        auto size = bin.size();
        bin.resize(size + count);
        MPI_Recv(&bin[size], count, PARTICLE, g.neignbors[dir], dir, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    };
    // Send outer boundary
    g.outer(send);
    // Receive inner boundary
    g.inner(receive);
    // Wait for all sends
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    bn = ceil(size / cutoff);
    gs = floor(sqrt(num_procs));
    gw = gs + (num_procs - gs * gs) / gs;
    if (rank >= gw * gs) return;
    g = Group(rank);
    for (int pi = 0; pi < num_parts; pi++) {
        particle_t p = parts[pi];
        p.ax = p.ay = 0;
        int bi = floor(p.x / cutoff) - g.i;
        int bj = floor(p.y / cutoff) - g.j;
        if (g.contains(bi, bj)) g.bin(bi, bj).push_back(p);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if (rank >= gw * gs) return;
    transmit_edges();
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < g.m; i++) {
        for (int j = 0; j < g.n; j++) {
            vector<particle_t>& bin = g.bin(i, j);
            for (particle_t& p : bin) {
                for (int ni : {i - 1, i, i + 1}) {
                    for (int nj : {j - 1, j, j + 1}) {
                        if (ni == i && nj == j) continue;
                        if (!g.contains(ni, nj)) {
                            for (particle_t& n : g.bin(ni, nj)) {
                                apply_force(p, n, false);
                            }
                        } else if (nj > j || (nj == j && ni > i)) {
                            for (particle_t& n : g.bin(ni, nj)) {
                                apply_force(p, n, true);
                            }
                        }
                    }
                }
            }
            for (int p = 0; p < bin.size(); p++) {
                for (int n = p + 1; n < bin.size(); n++) {
                    apply_force(bin[p], bin[n], true);
                }
            }
        }
    }
    struct movement {
        particle_t p;
        int i, j;
    };
    vector<movement> inter_moves;
    for (int i = -1; i <= g.m; i++) {
        g.bin(i, -1).clear();
        g.bin(i, g.n).clear();
    }
    for (int j = -1; j <= g.n; j++) {
        g.bin(-1, j).clear();
        g.bin(g.m, j).clear();
    }
    for (int i = 0; i < g.m; i++) {
        for (int j = 0; j < g.n; j++) {
            vector<particle_t>& bin = g.bin(i, j);
            auto f = [i, j, size, &inter_moves](particle_t& p){
                move(p, size);
                p.ax = p.ay = 0;
                int bi = floor(p.x / cutoff) - g.i;
                int bj = floor(p.y / cutoff) - g.j;
                if (bi == i && bj == j) return false;
                if (g.contains(bi, bj)) {
                    inter_moves.push_back({p, bi, bj});
                } else {
                    g.bin(bi, bj).push_back(p);
                }
                return true;
            };
            bin.erase(std::remove_if(bin.begin(), bin.end(), f), bin.end());
        }
    }
    for (movement m : inter_moves) {
        g.bin(m.i, m.j).push_back(m.p);
    }
    auto stop_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = stop_time - start_time;
    comp_time += diff.count();
    transmit_edges2();
}

void final_simulation(int rank) {
    printf("Rank %i computation time: %f\n", rank, comp_time);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if (rank >= gw * gs) return;
    vector<MPI_Request> requests;
    requests.reserve(g.m * g.n);
    for (int i = 0; i < g.m; i++) {
        for (int j = 0; j < g.n; j++) {
            vector<particle_t>& bin = g.bin(i, j);
            requests.emplace_back(MPI_Request());
            MPI_Isend(bin.data(), bin.size(), PARTICLE, 0, 0, MPI_COMM_WORLD, &requests.back());
        }
    }
    if (rank == 0) {
        particle_t* ps = parts;
        for (int i = 0; i < bn * bn; i++) {
            MPI_Status status;
            MPI_Recv(ps, num_parts, PARTICLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, PARTICLE, &count);
            ps += count;
        }
        std::sort(parts, parts + num_parts, [](particle_t const &a, particle_t const &b) {
            return a.id < b.id; 
        });
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}