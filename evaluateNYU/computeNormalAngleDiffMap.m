function thetaMap = computeNormalAngleDiffMap(predictedNormalMap, gtData)
    %unpack the ground truth
    gtNormalMap = zeros(size(predictedNormalMap));
    gtNormalMap(:,:,1) = gtData.nx;
    gtNormalMap(:,:,2) = gtData.ny;
    gtNormalMap(:,:,3) = gtData.nz;

    %normalize to be sure
    gtNormalDiv = sum(gtNormalMap.^2,3).^0.5;
    gtNormalMap = gtNormalMap ./ repmat(gtNormalDiv+eps,[1,1,3]);

    %compute the dot-product and then the acos
    dpMap = sum(gtNormalMap .* predictedNormalMap,3);
    dpMap = max(min(dpMap,1),-1);
    thetaMap = acos(dpMap);

end


