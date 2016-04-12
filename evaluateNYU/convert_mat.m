src = '/home/yu/seg_proj/FCN/Sec_2/test_res';

list = dir([src '/*.txt']); 

matsize = 32;
N3 = zeros(matsize, matsize, 3);

for i =1 : numel(list)
	fname = list(i).name;
	txtname = [src '/' fname];
	matname = strrep(fname, '.jpg.txt', '.mat'); 
	matname = [src '/' matname];
	fid = fopen(txtname, 'r');
	N3(:) = fscanf(fid, '%f'); 
	save(matname, 'N3');

	fclose(fid);

end



