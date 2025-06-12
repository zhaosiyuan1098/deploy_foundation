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
    explicit BlockQueue<T>(size_t max_size) ;
    bool blockPush(const T& item);
    bool coverPush(const T& item);
    std::optional<T> blockPop();
    std::optional<T> tryPop();
    void disablePush();
    void disablePop();
    void enablePush();
    void enablePop();
    int size();
    bool empty();
    void setNoMoreInput();
    ~BlockQueue() ;
    void DisableAndClear();


protected:

private:
    const size_t max_size_;
    std::queue<T> queue_;
    std::atomic<bool> push_enabled_;
    std::atomic<bool> pop_enabled_;
    std::condition_variable producer_cv_;
    std::condition_variable consumer_cv_;
    std::mutex mutex_;
    std::atomic<bool> no_more_input_;
};

template <typename T>
BlockQueue<T>::BlockQueue(const size_t max_size) :max_size_(max_size)
{
    push_enabled_.store(true);
    pop_enabled_.store(true);
    no_more_input_.store(false);
}

template <typename T>
bool BlockQueue<T>::blockPush(const T& item)
{
    std::unique_lock<std::mutex> lock(this->mutex_);
    while (queue_.size() >= max_size_&& push_enabled_.load())
    {
        this->producer_cv_.wait(lock);
    }
    if (!push_enabled_.load())
    {
        return false;
    }
    this->queue_.push(item);
    this->consumer_cv_.notify_one();
    return true;
}

template <typename T>
bool BlockQueue<T>::coverPush(const T& item)
{
    std::unique_lock<std::mutex> lock(this->mutex_);
    if (!push_enabled_.load())
    {
        return false;
    }
    if (queue_.size() == max_size_)
    {
        queue_.pop();
    }
    queue_.push(item);
    consumer_cv_.notify_one();
    return true;
}

template <typename T>
std::optional<T> BlockQueue<T>::blockPop()
{
    std::unique_lock<std::mutex> lock(this->mutex_);
    while (queue_.size()==0 && pop_enabled_ && !no_more_input_.load() )
    {
        consumer_cv_.wait(lock);
    }
    if (!pop_enabled_ || (no_more_input_ && queue_.size() == 0))
    {
        return std::nullopt;
    }
    T ans=queue_.front();
    queue_.pop();
    producer_cv_.notify_one();
    if (no_more_input_)
    {
        consumer_cv_.notify_all();
    }
    return ans;
}

template <typename T>
std::optional<T> BlockQueue<T>::tryPop()
{
    std::unique_lock<std::mutex> lock(this->mutex_);
    if (queue_.size() == 0)
    {
        return std::nullopt;
    }
    else
    {
        T ans =queue_.front();
        queue_.pop();
        producer_cv_.notify_all();
        if (no_more_input_)
        {
            consumer_cv_.notify_all();
        }
        return ans;
    }
}

template <typename T>
void BlockQueue<T>::disablePush()
{
    push_enabled_.store(false);
    producer_cv_.notify_all();
}

template <typename T>
void BlockQueue<T>::disablePop()
{
    pop_enabled_.store(false);
    consumer_cv_.notify_all();
}

template <typename T>
void BlockQueue<T>::enablePush()
{
    push_enabled_.store(true);
}

template <typename T>
void BlockQueue<T>::enablePop()
{
    pop_enabled_.store(true);
}

template <typename T>
int BlockQueue<T>::size()
{
    std::unique_lock<std::mutex> lock(this->mutex_);
    return this->queue_.size();
}

template <typename T>
bool BlockQueue<T>::empty()
{
    std::unique_lock<std::mutex> lock(this->mutex_);
    return this->size()==0;
}

template <typename T>
void BlockQueue<T>::setNoMoreInput()
{
    this->no_more_input_ = true;
    consumer_cv_.notify_all();
}

template <typename T>
BlockQueue<T>::~BlockQueue()
{
    disablePush();
    disablePop();
}

template <typename T>
void BlockQueue<T>::DisableAndClear()
{
    disablePush();
    disablePop();
    std::unique_lock<std::mutex> u_lck(mutex_);
    while (!queue_.empty()) queue_.pop();
}


#endif //BLOCKQUEUE_H
