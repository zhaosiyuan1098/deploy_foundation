//
// Created by root on 25-6-12.
//

#ifndef INFER_CORE_FACTORY_H
#define INFER_CORE_FACTORY_H

#include "infer_core.h"
class BaseInferCoreFactory
{
public:
    virtual std::shared_ptr<infer_core::BaseInferCore> Create() = 0;
    virtual ~BaseInferCoreFactory() = default;
};


#endif //INFER_CORE_FACTORY_H
