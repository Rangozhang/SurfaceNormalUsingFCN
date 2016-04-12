function D = decrop(S)
    SSize = size(S);
    D = zeros([480,640,SSize(3:end)]);
    for c=1:size(S,3)
        Dc = zeros(480,640);
        Dc(45:471,41:601) = S(:,:,c);
        D(:,:,c) = Dc;
    end
end
