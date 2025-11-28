#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & normal_(at::Tensor & self, double mean, double std, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(self.device(),rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_normal,mean,std*std,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */