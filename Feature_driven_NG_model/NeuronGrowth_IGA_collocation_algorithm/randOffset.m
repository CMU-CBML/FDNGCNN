function [val] = randOffset(X)
    val = round(rand(1)*X-X/2);
end