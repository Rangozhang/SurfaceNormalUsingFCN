function Nr = resizeNormals(N,sz)
    Nr = zeros([sz,3]);
    for c=1:3, Nr(:,:,c) = imresize_old(N(:,:,c),sz,'bilinear'); end
    norm = sum(Nr.^2,3).^0.5;
    Nr = bsxfun(@rdivide,Nr,norm+eps);
end
