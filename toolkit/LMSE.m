function [ score ] = LMSE( gt_shad, shad, mask, gt_ref, ref,window_size)

if (nargin <6)
    window_size = 20;
end
[ROW, COL, CHAN] = size(gt_shad);
score = 0;
for chan_iter = 1:CHAN
    score = score +1*local_error(gt_shad(:, :, chan_iter), shad(:, :, chan_iter), mask, window_size, floor(window_size/2));%+ ...
    %    0.5*local_error(gt_ref, ref, mask, window_size, floor(window_size/2));
end
score = score / CHAN;

function Variable = local_error(correct, estimate, mask, window_size, window_shift)

[M, N] = size(correct);
ssq = 0; total = 0;
for i = 1: window_shift: M-window_size
    for j = 1: window_shift: N-window_size
        correct_curr = correct(i:i+window_size, j:j+window_size);
        estimate_curr = estimate(i:i+window_size, j:j+window_size);
        mask_curr = mask(i:i+window_size, j:j+window_size);
        ssq = ssq +ssq_error(correct_curr, estimate_curr, mask_curr);
        curr = mask_curr .*(correct_curr.^2);
        total = total +sum(curr(:));
    end
end
Variable = ssq/(total+eps);


function Variable = ssq_error(correct, estimate, mask)

curr = estimate.^2.*mask;
if sum(curr(:)) > 1e-5
    alpha = sum(sum(correct.*estimate.*mask))/sum(curr(:));
else
    alpha =0;
end

Variable = sum(sum(mask.*((correct-alpha*estimate).^2)));
