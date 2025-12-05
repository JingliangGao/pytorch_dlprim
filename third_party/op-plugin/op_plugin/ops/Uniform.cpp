#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & uniform_(at::Tensor & self, double from, double to, c10::optional<Generator> generator)
    {

        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(self.device(),rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_uniform,from,to,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
