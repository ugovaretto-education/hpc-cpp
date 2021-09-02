#include <chrono>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>
#include <algorithm>
#include <ranges>

using namespace std;
using namespace chrono;
using namespace ranges;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "usage: " << argv[0] << " <number of time points>" << endl;
        exit(EXIT_FAILURE);
    }
    using T = decltype(high_resolution_clock::now());
    vector<T> ticks(stoul(argv[1]));
    vector<double> diffs;
    diffs.reserve(ticks.size());
    double mindist = numeric_limits<double>::max();
    double maxdist = 0.;
    for (auto& i : ticks) i = high_resolution_clock::now();
    for (auto n = ++begin(ticks); n != end(ticks); ++n) {
        const double d = duration<double>(*n - *(n - 1)).count();
        mindist = min(mindist, d);
        maxdist = max(maxdist, d);
        diffs.push_back(d);
    }
    sort(begin(diffs), end(diffs));
    cout << "Min distance:    " << mindist << endl
         << "Max distance:    " << maxdist << endl;
    cout << "Median distance: " << diffs[diffs.size() / 2] << endl;
    ranges::copy(diffs, std::ostream_iterator<double>(std::cout, "\n"));
    return 0;
}
