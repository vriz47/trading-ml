/* LSTM forward kernel (batch_first), row-major float32, 1 lapisan.
 * gates = x_t @ W_ih^T + (b_ih+b_hh) + h @ W_hh^T
 * i=sigmoid, f=sigmoid, g=tanh, o=sigmoid
 * c = f*c + i*g ; h = o*tanh(c)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float _sig(float x) { return 1.0f / (1.0f + expf(-x)); }

void lstm_fwd(const float *x, const float *wih, const float *whh,
              const float *bih, const float *bhh, float *hout,
              size_t B, size_t L, size_t F, size_t H)
{
    const size_t G = 4;
    const size_t rank0 = F;
    const size_t rank1 = H;
    float *biasc = (float *)malloc(G * rank1 * sizeof(float));
    for (size_t r = 0; r < G * rank1; r++) biasc[r] = bih[r] + bhh[r];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t b = 0; b < (ptrdiff_t)B; b++) {
        const float *xb = x + b * L * rank0;
        float *gates = (float *)malloc(G * rank1 * sizeof(float));
        float *c = (float *)calloc(rank1, sizeof(float));
        float *h = (float *)calloc(rank1, sizeof(float));

        for (size_t t = 0; t < L; t++) {
            const float *xt = xb + t * rank0;
            for (size_t r = 0; r < G * rank1; r++) {
                float acc = biasc[r];
                const float *wr = wih + r * rank0;
                for (size_t k = 0; k < rank0; k++) acc += xt[k] * wr[k];
                const float *wd = whh + r * rank1;
                for (size_t k = 0; k < rank1; k++) acc += h[k] * wd[k];
                gates[r] = acc;
            }
            for (size_t j = 0; j < rank1; j++) {
                float sg = _sig(gates[j]);
                float f = _sig(gates[rank1 + j]);
                float g = tanhf(gates[2 * rank1 + j]);
                float o = _sig(gates[3 * rank1 + j]);
                c[j] = f * c[j] + sg * g;
                h[j] = o * tanhf(c[j]);
            }
        }
        for (size_t j = 0; j < rank1; j++) hout[b * rank1 + j] = h[j];

        free(gates);
        free(c);
        free(h);
    }

    free(biasc);
}