### summarization (rougeL, dense = 26.92, n = 300)

| method | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|
| gradient | 26.64 (99%) | 26.21 (97%) | 24.60 (91%) | 23.40 (87%) | 20.53 (76%) | 15.70 (58%) |
| gradient_oneshot | 26.64 (99%) | 25.46 (95%) | 24.70 (92%) | 22.48 (83%) | 16.94 (63%) | 13.29 (49%) |
| magnitude | 26.42 (98%) | 25.62 (95%) | 22.74 (84%) | 17.87 (66%) | 13.05 (48%) | 11.35 (42%) |
| random | 24.90 (93%) | 23.77 (88%) | 22.53 (84%) | 20.56 (76%) | 17.76 (66%) | 14.38 (53%) |

### translation (bleu, dense = 25.49, n = 500)

| method | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|
| gradient | 25.16 (99%) | 24.88 (98%) | 24.08 (94%) | 22.81 (89%) | 20.04 (79%) | 9.50 (37%) |
| gradient_oneshot | 25.16 (99%) | 25.19 (99%) | 23.60 (93%) | 19.81 (78%) | 12.33 (48%) | 3.52 (14%) |
| magnitude | 25.24 (99%) | 23.97 (94%) | 19.22 (75%) | 7.39 (29%) | 1.76 (7%) | 0.39 (2%) |
| random | 24.96 (98%) | 23.88 (94%) | 22.58 (89%) | 15.07 (59%) | 8.22 (32%) | 1.03 (4%) |

### Efficiency — summarization on Tesla T4 (torch.float32)

| model | params (M) | attn params (M) | GFLOPs | peak GPU MB (bs16) | bs1 latency (ms) / speed-up | bs16 latency (ms) / speed-up |
|---|---|---|---|---|---|---|
| dense | 60.5 | 18.9 | 10.91 | 328 | 583.1 / 1.00x | 598.1 / 1.00x |
| masked 40% (no removal) | 60.5 | 18.9 | 10.91 | 328 | 632.8 / 0.92x | 648.5 / 0.92x |
| pruned 40% | 52.9 | 11.3 | 9.44 | 269 | 612.0 / 0.95x | 643.4 / 0.93x |

### Efficiency — translation on Tesla T4 (torch.float32)

| model | params (M) | attn params (M) | GFLOPs | peak GPU MB (bs16) | bs1 latency (ms) / speed-up | bs16 latency (ms) / speed-up |
|---|---|---|---|---|---|---|
| dense | 60.5 | 18.9 | 10.91 | 328 | 641.5 / 1.00x | 661.1 / 1.00x |
| masked 60% (no removal) | 60.5 | 18.9 | 10.91 | 328 | 701.4 / 0.91x | 703.0 / 0.94x |
| pruned 10% | 58.7 | 17.0 | 10.49 | 318 | 663.6 / 0.97x | 672.6 / 0.98x |
| pruned 20% | 56.7 | 15.1 | 10.09 | 303 | 658.8 / 0.97x | 678.7 / 0.97x |
| pruned 30% | 54.9 | 13.2 | 9.75 | 288 | 665.5 / 0.96x | 686.8 / 0.96x |
| pruned 40% | 52.9 | 11.3 | 9.35 | 274 | 675.9 / 0.95x | 687.4 / 0.96x |
| pruned 50% | 51.1 | 9.4 | 8.94 | 260 | 670.4 / 0.96x | 705.7 / 0.94x |
| pruned 60% | 49.2 | 7.6 | 8.50 | 250 | 665.6 / 0.96x | 692.6 / 0.95x |

### Recovery fine-tuning — summarization (rougeL; 8000 train examples, 1 epoch, lr 0.0001)

| model | after pruning | after fine-tuning | % of dense | % of dense + same FT |
|---|---|---|---|---|
| dense | 26.92 | 28.89 | 107% | 100% |
| pruned 40% | 23.40 | 27.31 | 101% | 95% |
| pruned 50% | 20.53 | 27.13 | 101% | 94% |
| pruned 60% | 15.70 | 25.92 | 96% | 90% |

### Recovery fine-tuning — translation (bleu; 20000 train examples, 1 epoch, lr 0.0001)

| model | after pruning | after fine-tuning | % of dense | % of dense + same FT |
|---|---|---|---|---|
| dense | 25.49 | 25.80 | 101% | 100% |
| pruned 40% | 22.81 | 24.80 | 97% | 96% |
| pruned 50% | 20.04 | 23.45 | 92% | 91% |
| pruned 60% | 9.50 | 21.74 | 85% | 84% |
