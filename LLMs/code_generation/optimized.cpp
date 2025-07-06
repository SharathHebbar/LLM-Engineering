
#include <iostream>
#include <iomanip>
#include <cmath>

double calculate(long long iterations, double param1, double param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }
    return result;
}

int main() {
    long long iterations = 100000000LL;
    double param1 = 4.0;
    double param2 = 1.0;
    double result = calculate(iterations, param1, param2) * 4.0;
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << std::endl;
    std::cout << "Execution Time: " << (time_point_cast<chrono::seconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count() - time_point_cast<chrono::seconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count()) / 1000000.0 << " seconds" << std::endl;
    return 0;
}
