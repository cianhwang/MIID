function [data rgbImg height width numSpec wl]=readDat(fileName)

fp=fopen(fileName,'rb+');

height=fread(fp,1,'int');
width=fread(fp,1,'int');
numSpec=fread(fp,1,'int');

wl=fread(fp,numSpec,'float');

rgbImage1 = fread(fp,height*width*3,'float');
rgbImage2 = reshape(rgbImage1,[3 width height]);
rgbImage3=permute(rgbImage2,[3 2 1]);
rgbImg = rgbImage3;

data1=fread(fp,height*width*numSpec,'float');

data2=reshape(data1,[numSpec width height]);
data3=permute(data2,[3 2 1]);

data=data3;

fclose(fp);

end

