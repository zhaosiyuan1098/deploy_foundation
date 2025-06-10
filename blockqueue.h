//
// Created by siyuan on 25-6-10.
//

#ifndef BLOCKQUEUE_H
#define BLOCKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <optional>
#include <queue>

template <typename T>
class BlockQueue
{
public:
    BlockQueue<T>() ;
    bool blockPush();
    bool coverPush();
    std::optional<T> blockPop();
    std::optional<T> tryPop();
    bool disablePush();
    bool disablePop();
    bool enablePush();
    bool enablePop();
    int size();
    bool empty();
protected:
    ~BlockQueue() ;
private:
    const size_t size_;
    std::queue<T> queue_;
    std::atomic<bool> push_enabled_;
    std::atomic<bool> pop_enabled_;
    std::condition_variable producer_cv_;
    std::condition_variable consumer_cv_;
    std::mutex mutex_;
};

template <typename T>
BlockQueue<T>::BlockQueue()
{

}



#endif //BLOCKQUEUE_H
