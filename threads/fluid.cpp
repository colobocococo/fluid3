#include <bits/stdc++.h>

using namespace std;

constexpr size_t N = 36, M = 84 + 1e5;
// constexpr size_t N = 14, M = 5;
constexpr size_t T = 30;
constexpr std::array<pair<int, int>, 4> deltas{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

// char field[N][M + 1] = {
//     "#####",
//     "#.  #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#...#",
//     "#####",
//     "#   #",
//     "#   #",
//     "#   #",
//     "#####",
// };

string field[N] = {
    "####################################################################################",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                       .........                                  #",
    "#..............#            #           .........                                  #",
    "#..............#            #           .........                                  #",
    "#..............#            #           .........                                  #",
    "#..............#            #                                                      #",
    "#..............#            #                                                      #",
    "#..............#            #                                                      #",
    "#..............#            #                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............################                     #                 #",
    "#...........................#....................................#                 #",
    "#...........................#....................................#                 #",
    "#...........................#....................................#                 #",
    "##################################################################                 #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "####################################################################################",
};

void start() {
    for (int i = 0; i < N; i++)
        for (int j = 84; j < M; j++) field[i] += '#';
}

struct Fixed {
    constexpr Fixed(int v): v(v << 16) {}
    constexpr Fixed(float f): v(f * (1 << 16)) {}
    constexpr Fixed(double f): v(f * (1 << 16)) {}
    constexpr Fixed(): v(0) {}

    static constexpr Fixed from_raw(int32_t x) {
        Fixed ret;
        ret.v = x;
        return ret;
    } 

    int32_t v;

    auto operator<=>(const Fixed&) const = default;
    bool operator==(const Fixed&) const = default;
};

static constexpr Fixed inf = Fixed::from_raw(std::numeric_limits<int32_t>::max());
static constexpr Fixed eps = Fixed::from_raw(deltas.size());

Fixed operator+(Fixed a, Fixed b) {
    return Fixed::from_raw(a.v + b.v);
}

Fixed operator-(Fixed a, Fixed b) {
    return Fixed::from_raw(a.v - b.v);
}

Fixed operator*(Fixed a, Fixed b) {
    return Fixed::from_raw(((int64_t) a.v * b.v) >> 16);
}

Fixed operator/(Fixed a, Fixed b) {
    return Fixed::from_raw(((int64_t) a.v << 16) / b.v);
}

Fixed &operator+=(Fixed &a, Fixed b) {
    return a = a + b;
}

Fixed &operator-=(Fixed &a, Fixed b) {
    return a = a - b;
}

Fixed &operator*=(Fixed &a, Fixed b) {
    return a = a * b;
}

Fixed &operator/=(Fixed &a, Fixed b) {
    return a = a / b;
}

Fixed operator-(Fixed x) {
    return Fixed::from_raw(-x.v);
}

Fixed abs(Fixed x) {
    if (x.v < 0) {
        x.v = -x.v;
    }
    return x;
}

ostream &operator<<(ostream &out, Fixed x) {
    return out << x.v / (double) (1 << 16);
}

Fixed rho[256];

Fixed p[N][M]{}, old_p[N][M];

struct VectorField {
    array<Fixed, deltas.size()> v[N][M];
    Fixed &add(int x, int y, int dx, int dy, Fixed dv) {
        return get(x, y, dx, dy) += dv;
    }

    Fixed &get(int x, int y, int dx, int dy) {
        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
        assert(i < deltas.size());
        return v[x][y][i];
    }
};

VectorField velocity{}, velocity_flow{};
int last_use[N][M]{};
int UT = 0;


mt19937 rnd(1337);
mutex h;

tuple<Fixed, bool, pair<int, int>> propagate_flow(int x, int y, Fixed lim) {
    h.lock();
    last_use[x][y] = UT - 1;
    h.unlock();
    Fixed ret = 0;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT) {
            h.lock();
            auto cap = velocity.get(x, y, dx, dy); //global
            auto flow = velocity_flow.get(x, y, dx, dy); //global
            h.unlock();
            if (flow == cap) {
                continue;
            }
            // assert(v >= velocity_flow.get(x, y, dx, dy));
            auto vp = min(lim, cap - flow);
            if (last_use[nx][ny] == UT - 1) {
                h.lock();
                velocity_flow.add(x, y, dx, dy, vp);
                last_use[x][y] = UT;
                h.unlock();
                // cerr << x << " " << y << " -> " << nx << " " << ny << " " << vp << " / " << lim << "\n";
                return {vp, 1, {nx, ny}};
            }
            auto [t, prop, end] = propagate_flow(nx, ny, vp);
            ret += t;
            if (prop) {
                h.lock();
                velocity_flow.add(x, y, dx, dy, t);
                last_use[x][y] = UT;
                h.unlock();
                // cerr << x << " " << y << " -> " << nx << " " << ny << " " << t << " / " << lim << "\n";
                return {t, prop && end != pair(x, y), end};
            }
        }
    }
    h.lock();
    last_use[x][y] = UT;
    h.unlock();
    return {ret, 0, {0, 0}};
}

Fixed random01() {
    return Fixed::from_raw((rnd() & ((1 << 16) - 1)));
}

void propagate_stop(int x, int y, bool force = false) {
    if (!force) {
        bool stop = true;
        for (auto [dx, dy] : deltas) {
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) > 0) {
                stop = false;
                break;
            }
        }
        if (!stop) {
            return;
        }
    }
    last_use[x][y] = UT;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT || velocity.get(x, y, dx, dy) > 0) {
            continue;
        }
        //thread a(propagate_stop, nx, ny, false);
        //a.detach();
        propagate_stop(nx, ny);
    }
}

Fixed move_prob(int x, int y) {
    Fixed sum = 0;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
            continue;
        }
        auto v = velocity.get(x, y, dx, dy);
        if (v < 0) {
            continue;
        }
        sum += v;
    }
    return sum;
}

struct ParticleParams {
    char type;
    Fixed cur_p;
    array<Fixed, deltas.size()> v;

    void swap_with(int x, int y) {
        swap(field[x][y], type);
        swap(p[x][y], cur_p);
        swap(velocity.v[x][y], v);
    }
};

bool propagate_move(int x, int y, bool is_first) {
    last_use[x][y] = UT - is_first;
    bool ret = false;
    int nx = -1, ny = -1;
    do {
        std::array<Fixed, deltas.size()> tres;
        Fixed sum = 0;
        for (size_t i = 0; i < deltas.size(); ++i) {
            auto [dx, dy] = deltas[i];
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
                tres[i] = sum;
                continue;
            }
            auto v = velocity.get(x, y, dx, dy);
            if (v < 0) {
                tres[i] = sum;
                continue;
            }
            sum += v;
            tres[i] = sum;
        }

        if (sum == 0) {
            break;
        }

        Fixed p = random01() * sum;
        size_t d = std::ranges::upper_bound(tres, p) - tres.begin();

        auto [dx, dy] = deltas[d];
        nx = x + dx;
        ny = y + dy;
        assert(velocity.get(x, y, dx, dy) > 0 && field[nx][ny] != '#' && last_use[nx][ny] < UT);

        ret = (last_use[nx][ny] == UT - 1 || propagate_move(nx, ny, false));
    } while (!ret);
    last_use[x][y] = UT;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) < 0) {
            propagate_stop(nx, ny);
        }
    }
    if (ret) {
        if (!is_first) {
            ParticleParams pp{};
            pp.swap_with(x, y);
            pp.swap_with(nx, ny);
            pp.swap_with(x, y);
        }
    }
    return ret;
}

int dirs[N][M]{};
Fixed total_delta_p; //global, need mutex

mutex m;
int n_threads = 1000;
Fixed g = 0.1;
bool prop;

void f1(size_t l, size_t r) {
    while (l < r) {
        size_t x = l/M, y = l%M;
        if (field[x][y] == '#') {
            l++;
            continue;
        }
        for (auto [dx, dy] : deltas) {
            dirs[x][y] += (field[x + dx][y + dy] != '#');
        }
        l++;
    }
}

void f2(size_t l, size_t r) {
    while (l < r) {
        size_t x = l/M, y = l%M;
        if (field[x][y] == '#') {
            l++;
            continue;
        }
        for (auto [dx, dy] : deltas) {
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] != '#' && old_p[nx][ny] < old_p[x][y]) {
                auto delta_p = old_p[x][y] - old_p[nx][ny];
                auto force = delta_p;
                auto &contr = velocity.get(nx, ny, -dx, -dy);
                if (contr * rho[(int) field[nx][ny]] >= force) {
                    contr -= force / rho[(int) field[nx][ny]];
                    continue;
                }
                force -= contr * rho[(int) field[nx][ny]];
                contr = 0;
                p[x][y] -= force / dirs[x][y];
                m.lock();
                velocity.add(x, y, dx, dy, force / rho[(int) field[x][y]]); //global
                total_delta_p -= force / dirs[x][y]; //must be a mutex
                m.unlock();
            }
        }
        l++;
    }
}

void f3(size_t l, size_t r) {
    while (l < r) {
        size_t x = l/M, y = l%M;
        if (field[x][y] == '#') {
            l++;
            continue;
        }
        for (auto [dx, dy] : deltas) {
            m.lock();
            auto old_v = velocity.get(x, y, dx, dy);
            auto new_v = velocity_flow.get(x, y, dx, dy);
            if (old_v > 0) {
                assert(new_v <= old_v);
                velocity.get(x, y, dx, dy) = new_v;
                auto force = (old_v - new_v) * rho[(int) field[x][y]];
                if (field[x][y] == '.')
                    force *= 0.8;
                if (field[x + dx][y + dy] == '#') {
                    p[x][y] += force / dirs[x][y];
                    total_delta_p += force / dirs[x][y];
                } else {
                    p[x + dx][y + dy] += force / dirs[x + dx][y + dy];
                    total_delta_p += force / dirs[x + dx][y + dy];
                }
            }
            m.unlock();
        }
        l++;
    }
}

void f4(size_t l, size_t r) {
    while (l < r) {
        size_t x = l/M, y = l%M;
        if (field[x][y] == '#') {
            l++;
            continue;
        }
        if (field[x + 1][y] != '#') {
            m.lock();
            velocity.add(x, y, 1, 0, g);
            m.unlock();
        }
        l++;
        //cout << l;
    }
}

void f5(size_t l, size_t r) {
    //m.lock();
    while (l < r) {
        size_t x = l/84, y = l%84;
        if (field[x][y] != '#' && last_use[x][y] < UT - 1) {
            //m.lock();
            auto [t, local_prop, _] = propagate_flow(x, y, 1);
            //m.unlock();
            if (t > 0) {
                m.lock();
                prop = 1;
                m.unlock();
            }
        }
        l++;
    }
    //m.unlock();
}

void f6(size_t l, size_t r) {
    //m.lock();
    while (l < r) {
        size_t x = l/M, y = l%M;
        if (y < 84) {
            l++;
            continue;
        }
        //cout << M << ' ';
        if (field[x][y] != '#' && last_use[x][y] < UT - 1) {
            //m.lock();
            auto [t, local_prop, _] = propagate_flow(x, y, 1);
            //m.unlock();
            if (t > 0) {
                m.lock();
                prop = 1;
                m.unlock();
            }
        }
        l++;
    }
    //m.unlock();
}

int main() {
    start();
    cin >> n_threads; //количество нитей
    size_t piece = N*M/n_threads;
    rho[' '] = 0.01;
    rho['.'] = 1000;

    //cout << clock() << "\n";

    int tm = 0, tm1 = 0, tm2 = 0, tm3 = 0, tm4 = 0, tm5 = 0;
    tm1 -= clock();
    for (int k = n_threads - 1; k >= 0; k--) {
        size_t l = k*piece, r = min(N*M, (k+1)*piece);
        //f1(l, r);
        thread a(f1, l, r);
        if (k) a.detach();
        else a.join();
    }
    tm1 += clock();


    tm -= clock();
    for (size_t i = 0; i < T; ++i) {
        
        total_delta_p = 0;
        // Apply external forces
        tm4 -= clock();
        for (int k = n_threads - 1; k >= 0; k--) {
            size_t l = k*piece, r = min(N*M, (k+1)*piece);
            //f4(l, r);
            thread a(f4, l, r);
            if (k) a.detach();
            else a.join();
        }
        tm4 += clock();

        // Apply forces from p
        memcpy(old_p, p, sizeof(p));
        tm2 -= clock();
        for (int k = n_threads - 1; k >= 0; k--) {
            size_t l = k*piece, r = min(N*M, (k+1)*piece);
            //f2(l, r);
            thread a(f2, l, r);
            if (k) a.detach();
            else a.join();
        }
        tm2 += clock();

        // Make flow from velocities
        velocity_flow = {};
        tm5 -= clock();
        prop = false;
        do {
            UT += 2;
            prop = 0;
            for (int k = n_threads - 1; k >= 0; k--) {
                size_t l = k*piece, r = min(N*M, (k+1)*piece);
                r = N*M - l;
                l = max(r, piece) - piece;
                //f6(l, r);
                thread a(f6, l, r);
                if (k) a.detach();
                else a.join();
            }
            f5(0, N*84);
            //this_thread::sleep_for(100ms);

            //this_thread::sleep_for(1000ms);
        } while (prop);
        tm5 += clock();
        // Recalculate p with kinetic energy
        tm3 -= clock();
        for (int k = n_threads - 1; k >= 0; k--) {
            size_t l = k*piece, r = min(N*M, (k+1)*piece);
            //f3(l, r);
            thread a(f3, l, r);
            if (k) a.detach();
            else a.join();
        }
        this_thread::sleep_for(1ms);
        tm3 += clock();

        UT += 2;
        prop = false;
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] != '#' && last_use[x][y] != UT) {
                    if (random01() < move_prob(x, y)) {
                        prop = true;
                        propagate_move(x, y, true);
                    } else {
                        propagate_stop(x, y, true);
                    }
                }
            }
        }

        //prop = false;
        if (prop) { //can't speed up this
            cout << "Tick " << i << ":\n";
            for (size_t x = 0; x < N; ++x) {
                //cout << field[x] << "\n";
            }
        }
    }

    tm += clock();
    cout << tm1 << ' ' << tm2 << ' ' << tm3 << ' ' << tm4 << ' ' << tm5;
    cout << "\n" << clock();
}
