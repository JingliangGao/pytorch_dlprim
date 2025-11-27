#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & bernoulli_(at::Tensor & self, double p, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(self.device(),rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_bernoulli,p,0,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */