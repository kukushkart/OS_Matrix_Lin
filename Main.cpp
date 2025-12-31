#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <iomanip>

using namespace std;
using namespace chrono;

const int N = 50; 

void initMatrix(vector<vector<double>>& matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

// однопоточное
void multiplySimple(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multiplyBlock(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int rowStart, int rowEnd,
    int colStart, int colEnd) {
    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void multiplyThreadStd(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize) {
    vector<thread> threads;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numBlocks; j++) {
            int rowStart = i * blockSize;
            int rowEnd = min((i + 1) * blockSize, N);
            int colStart = j * blockSize;
            int colEnd = min((j + 1) * blockSize, N);

            threads.emplace_back(multiplyBlock, ref(A), ref(B), ref(C),
                rowStart, rowEnd, colStart, colEnd);
        }
    }

    for (auto& t : threads) {
        t.join();
    }
}

struct BlockParams {
    const vector<vector<double>>* A;
    const vector<vector<double>>* B;
    vector<vector<double>>* C;
    int rowStart, rowEnd;
    int colStart, colEnd;
};

void* multiplyBlockPthread(void* param) {
    BlockParams* p = (BlockParams*)param;
    multiplyBlock(*(p->A), *(p->B), *(p->C),
        p->rowStart, p->rowEnd, p->colStart, p->colEnd);
    delete p;
    return nullptr;
}

void multiplyThreadPthread(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize) {
    vector<pthread_t> threads;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numBlocks; j++) {
            BlockParams* p = new BlockParams;
            p->A = &A;
            p->B = &B;
            p->C = &C;
            p->rowStart = i * blockSize;
            p->rowEnd = min((i + 1) * blockSize, N);
            p->colStart = j * blockSize;
            p->colEnd = min((j + 1) * blockSize, N);

            pthread_t thread;
            pthread_create(&thread, nullptr, multiplyBlockPthread, p);
            threads.push_back(thread);
        }
    }

    for (auto& t : threads) {
        pthread_join(t, nullptr);
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    srand(time(0));

    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N));

    initMatrix(A);
    initMatrix(B);

    cout << "\nОднопоточное умножение" << endl;
    auto start = high_resolution_clock::now();
    multiplySimple(A, B, C);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    cout << "Время: " << duration << " мкс" << endl;

    cout << left << setw(15) << "Размер блока"
        << setw(15) << "Кол-во блоков"
        << setw(20) << "std::thread (мкс)"
        << setw(20) << "pthread (мкс)" << endl;

    vector<int> blockSizes;
    for (int k = 1; k <= N; k *= 2) {
        blockSizes.push_back(k);
    }
    if (blockSizes.back() != N) {
        blockSizes.push_back(N);
    }

    for (int blockSize : blockSizes) {
        int numBlocks = (N + blockSize - 1) / blockSize;
        int totalBlocks = numBlocks * numBlocks;

        // std::thread
        start = high_resolution_clock::now();
        multiplyThreadStd(A, B, C, blockSize);
        end = high_resolution_clock::now();
        auto timeStd = duration_cast<microseconds>(end - start).count();

        // pthread
        start = high_resolution_clock::now();
        multiplyThreadPthread(A, B, C, blockSize);
        end = high_resolution_clock::now();
        auto timePthread = duration_cast<microseconds>(end - start).count();

        cout << left << setw(15) << blockSize
            << setw(15) << totalBlocks
            << setw(20) << timeStd
            << setw(20) << timePthread << endl;
    }

    cout << "\nТестирование завершено!" << endl;

    return 0;
}