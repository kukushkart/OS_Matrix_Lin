#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <iomanip>
#include <pthread.h>
#include <cstring>

using namespace std;
using namespace chrono;

const int N = 20;

void initMatrix(vector<vector<double>>& matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

void clearMatrix(vector<vector<double>>& matrix) {
    for (int i = 0; i < N; i++) {
        fill(matrix[i].begin(), matrix[i].end(), 0.0);
    }
}

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

void multiplyBlockKernel(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int rowStart, int rowEnd,
    int colStart, int colEnd,
    int innerStart, int innerEnd,
    mutex* mtx,
    pthread_mutex_t* p_mtx) {

    int h = rowEnd - rowStart;
    int w = colEnd - colStart;
    vector<vector<double>> localRes(h, vector<double>(w, 0.0));

    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            double sum = 0;
            for (int k = innerStart; k < innerEnd; k++) {
                sum += A[i][k] * B[k][j];
            }
            localRes[i - rowStart][j - colStart] = sum;
        }
    }

    if (mtx) mtx->lock();
    if (p_mtx) pthread_mutex_lock(p_mtx);

    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            C[i][j] += localRes[i - rowStart][j - colStart];
        }
    }

    if (p_mtx) pthread_mutex_unlock(p_mtx);
    if (mtx) mtx->unlock();
}

void multiplyThreadStd(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize) {

    vector<thread> threads;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vector<unique_ptr<mutex>> mutexes;
    for (int i = 0; i < numBlocks * numBlocks; ++i) {
        mutexes.push_back(make_unique<mutex>());
    }

    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numBlocks; j++) {
            for (int k = 0; k < numBlocks; k++) {

                int rowStart = i * blockSize;
                int rowEnd = min((i + 1) * blockSize, N);

                int colStart = j * blockSize;
                int colEnd = min((j + 1) * blockSize, N);

                int innerStart = k * blockSize;
                int innerEnd = min((k + 1) * blockSize, N);

                mutex* mtx = mutexes[i * numBlocks + j].get();

                threads.emplace_back(multiplyBlockKernel,
                    ref(A), ref(B), ref(C),
                    rowStart, rowEnd, colStart, colEnd, innerStart, innerEnd,
                    mtx, nullptr);
            }
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

struct BlockParams {
    const vector<vector<double>>* A;
    const vector<vector<double>>* B;
    vector<vector<double>>* C;
    int rowStart, rowEnd;
    int colStart, colEnd;
    int innerStart, innerEnd;
    pthread_mutex_t* p_mtx;
};

void* multiplyBlockPthreadWrapper(void* param) {
    BlockParams* p = (BlockParams*)param;
    multiplyBlockKernel(*(p->A), *(p->B), *(p->C),
        p->rowStart, p->rowEnd, p->colStart, p->colEnd,
        p->innerStart, p->innerEnd, nullptr, p->p_mtx);
    delete p;
    return nullptr;
}

void multiplyThreadPthread(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize) {

    vector<pthread_t> threads;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vector<pthread_mutex_t> mutexes(numBlocks * numBlocks);
    for (auto& m : mutexes) {
        pthread_mutex_init(&m, nullptr);
    }

    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numBlocks; j++) {
            for (int k = 0; k < numBlocks; k++) {

                BlockParams* p = new BlockParams;
                p->A = &A; p->B = &B; p->C = &C;
                p->rowStart = i * blockSize;
                p->rowEnd = min((i + 1) * blockSize, N);
                p->colStart = j * blockSize;
                p->colEnd = min((j + 1) * blockSize, N);
                p->innerStart = k * blockSize;
                p->innerEnd = min((k + 1) * blockSize, N);
                p->p_mtx = &mutexes[i * numBlocks + j];

                pthread_t threadId;
                if (pthread_create(&threadId, nullptr, multiplyBlockPthreadWrapper, p) == 0) {
                    threads.push_back(threadId);
                }
                else {
                    delete p;
                }
            }
        }
    }

    for (auto& t : threads) {
        pthread_join(t, nullptr);
    }

    for (auto& m : mutexes) {
        pthread_mutex_destroy(&m);
    }
}

int main() {
    srand((unsigned int)time(0));

    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N));

    initMatrix(A);
    initMatrix(B);

    cout << "Размер матрицы: " << N << "x" << N << endl;
    cout << "Однопоточное умножение " << endl;

    auto start = high_resolution_clock::now();
    multiplySimple(A, B, C);
    auto end = high_resolution_clock::now();
    auto durationSimple = duration_cast<milliseconds>(end - start).count();

    cout << "Время (один поток): " << durationSimple << " мс" << endl << endl;

    cout << left << setw(15) << "Размер блока"
        << setw(15) << "Кол-во потоков"
        << setw(20) << "std::thread (мс)"
        << setw(20) << "pthread (мс)"
        << endl;
    cout << endl;

    vector<int> blockSizes;
    for (int k = 1; k <= N; k *= 2) {
        blockSizes.push_back(k);
    }
    if (blockSizes.back() != N) {
        blockSizes.push_back(N);
    }

    for (int blockSize : blockSizes) {
        int numBlocks = (N + blockSize - 1) / blockSize;
        long long totalThreads = (long long)numBlocks * numBlocks * numBlocks;

        if (totalThreads > 5000) {
            cout << left << setw(15) << blockSize
                << setw(15) << totalThreads
                << setw(40) << "Слишком много потоков (skip)" << endl;
            continue;
        }

        clearMatrix(C);
        start = high_resolution_clock::now();
        multiplyThreadStd(A, B, C, blockSize);
        end = high_resolution_clock::now();
        auto timeStd = duration_cast<milliseconds>(end - start).count();

        clearMatrix(C);
        start = high_resolution_clock::now();
        multiplyThreadPthread(A, B, C, blockSize);
        end = high_resolution_clock::now();
        auto timePthread = duration_cast<milliseconds>(end - start).count();

        cout << left << setw(15) << blockSize
            << setw(15) << totalThreads
            << setw(20) << timeStd
            << setw(20) << timePthread
            << endl;
    }

    return 0;
}

