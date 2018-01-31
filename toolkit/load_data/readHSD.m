function[data,startw,endw,stepw]=readHSD(filename)
% readHSD Function for reading an HSD file
%
%       [data,startw,endw,stepw]=readHSD(filename)
%
%       This function can be used for reading an hyperspectral image file in HSD format.
%
%       filename = path of file (string)
%       data = hyperspectral image data (size: height*width*channel)
%       startw = the start wavelengths
%       endw = the end wavelengths
%       stepw = spectral resolution
%
%
%       [data,startw,endw,stepw]=readHSD('F:\sample.hsd')
%       reads contents of spectral image file F:\sample.hsd. 
%
% ------------------------------------------------------------


%fid=fopen('F:\SpectrumData.hsd','r');
fid=fopen(filename,'rb');
if fid<0
    disp('error data path');
    return;
end

height=fread(fid,1,'int');
width=fread(fid,1,'int');
SR=fread(fid,1,'int');
D=fread(fid,1,'int');
startw=fread(fid,1,'int');
stepw=fread(fid,1,'float');
endw=fread(fid,1,'int');

average=fread(fid,SR,'float');
coeff=fread(fid,D*SR,'float');
scoredata=fread(fid,height*width*D,'float');

coeff=reshape(coeff,SR,D);
scoredata=reshape(scoredata,D,height*width);
average=average';
coeff=coeff';
scoredata=scoredata';

fclose(fid);

temp=scoredata*coeff;
clear coeff;
clear scoredata;
data1=bsxfun(@plus,temp,average);
clear average;

data1=reshape(data1,width,height,SR);
for i=1:SR
    data(:,:,i)=data1(:,:,i)';
end
clear data1;



    
