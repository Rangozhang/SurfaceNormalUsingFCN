function gtData = loadGT(imageNum)
    if(1)
        %normalLocation = '/nfs/onega_no_backups1/users/dfouhey/NYUData/testNormals/';
        %  tv denoise
        normalLocation = '/home/yu/seg_proj/FCN/3d/testNormals/';
        gtData = load(sprintf('%s/nm_%06d.mat',normalLocation,imageNum)); 
    elseif(1)
        normalLocation = '/nfs/ladoga_no_backups/users/dfouhey/NYUData/normals/';    
        gtData = load(sprintf('%s/nm_%06d.mat',normalLocation,imageNum)); 
    else
        normalLocation = '/nfs/hn46/dfouhey/deepProcessedImageDS/data/test/';
        gtData = load(sprintf('%s/%08d_norm.mat',normalLocation,imageNum));
        gtData.depthValid = decrop((gtData.NMask > 0));
        gtData.nx = decrop(gtData.N(:,:,1));
        gtData.ny = decrop(gtData.N(:,:,2));
        gtData.nz = decrop(gtData.N(:,:,3));
    end
end
