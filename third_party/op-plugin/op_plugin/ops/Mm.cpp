#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

namespace {
    static at::Tensor get_bmm_mm_valid_tensor(at::Tensor const &t,bool &transposed,int &ld,bool &copied,int bmm,char /*M*/)
    {
        auto sizes= t.sizes();
        auto strides = t.strides();
        TORCH_CHECK(sizes.size() == 2u + bmm,"Invalid input matrix shape");
        TORCH_CHECK(sizes[0+bmm] > 0 && sizes[1+bmm] > 0,"Invalid matrix size");
        copied = false;
        if(t.is_contiguous())  {
            ld = strides[0+bmm];
            transposed = false;
            return t;
        }
        if(strides[1+bmm] >= sizes[0+bmm] && strides[0+bmm] == 1) {
            ld = strides[1+bmm];
            transposed = true;
            return t;
        }
        if(strides[0+bmm] >= sizes[1+bmm] && strides[1+bmm] == 1) {
            ld = strides[0+bmm];
            transposed = false;
            return t;
        }
        transposed = false;
        copied = true;
        ld = sizes[1+bmm];
        return t.contiguous();
    }

    at::Tensor & mm_out_nocheck(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out,int bmm)
    {

        at::Tensor A,B,C;
        bool At=false,Bt=false,Ct=false,Ac,Bc,Cc;
        int lda,ldb,ldc;
        A = get_bmm_mm_valid_tensor(self,At,lda,Ac,bmm,'A');
        B = get_bmm_mm_valid_tensor(mat2,Bt,ldb,Bc,bmm,'B');
        C = get_bmm_mm_valid_tensor(out, Ct,ldc,Cc,bmm,'C');
        if(Ct) {
            Ct = false;
            A=torch::transpose(A,0+bmm,1+bmm);
            B=torch::transpose(B,0+bmm,1+bmm);
            C=torch::transpose(C,0+bmm,1+bmm);
            At = !At;
            Bt = !Bt;
            std::swap(A,B);
            std::swap(lda,ldb);
            std::swap(At,Bt);
        }

        int M  = A.sizes()[0+bmm];
        int Ka = A.sizes()[1+bmm];
        int N  = B.sizes()[1+bmm];
        int Kb = B.sizes()[0+bmm];
        int Mc = C.sizes()[0+bmm];
        int Nc = C.sizes()[1+bmm];
        int K = Ka;

        TORCH_CHECK(M==Mc && N==Nc && Ka == Kb,"Invalid matrix sizes "
                    "A(" + std::to_string(M) + ","+std::to_string(Ka)+")" + (At?".T":"  ") +
                    "*B(" + std::to_string(Kb) + "," + std::to_string(N) +")=" + (Bt?".T":"  ") +
                    "C("+std::to_string(Mc) + ","+std::to_string(Nc)+")");

        TORCH_CHECK(A.dtype() == B.dtype() && A.dtype() == C.dtype(),"All matrices must have same dtype");
        if(bmm) {
            TORCH_CHECK(A.sizes()[0] == B.sizes()[0] && A.sizes()[0] == C.sizes()[0],"Matrices must have same batch i.e. 0 dimention");
        }


        dlprim::ExecutionContext q(getExecutionContext(self));
        dlprim::Context ctx(q);

        cl::Buffer Abuf = buffer_from_tensor(A);
        int64_t    Aoff = A.storage_offset();
        cl::Buffer Bbuf = buffer_from_tensor(B);
        int64_t    Boff = B.storage_offset();
        cl::Buffer Cbuf = buffer_from_tensor(C);
        int64_t    Coff = C.storage_offset();

        if(bmm == 0) {
            auto gemm_op = dlprim::gpu::GEMM::get_optimal_gemm(ctx,todp(A.dtype()),At,Bt,M,N,K);
            gemm_op->gemm(M,N,K,
                    Abuf,Aoff,lda,
                    Bbuf,Boff,ldb,
                    Cbuf,Coff,ldc,
                    nullptr,0,0,M*N,q);
        }
        else {
            int batch = A.sizes()[0];
            int step_A = A.strides()[0];
            int step_B = B.strides()[0];
            int step_C = C.strides()[0];
            dlprim::gpu::GEMM::batch_sgemm(todp(A.dtype()),
                At,Bt,
                batch,M,N,K,
                Abuf,Aoff,step_A,lda,
                Bbuf,Boff,step_B,ldb,
                Cbuf,Coff,step_C,ldc,
                0.0f,q);
        }
        if(Cc)
            out.copy_(C);
        sync_if_needed(self.device());
        return out;
    }

}

    at::Tensor & mm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out)
    {
        return mm_out_nocheck(self,mat2,out,0);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
