%Recover reflectance image from a color image
%Assuming the reflectance vectors have a low rank, with basis B so that
%R_p=B A_p
%Represent all unknowns consisting of A_p for all pixels in a vector A
%Using constant shading constraint when there is significant "hue" edge so
%that SH*A=0, SV*A=0
%Using smoothness constant where there is no significant "hue" change so
%that DH*A=0, DV*A=0;
%Further using boundary pixels reflectance are close to the color image so
%that  I*A=C, along different boundaries
%Solve A that minimizes L2 norm of the discrepancy from above assumptions
% Yao Wang,  11/10/2015

%using B determined from an image area of 100x100

clear all
close all

B =[

    0.5834   -0.7957
    0.5759    0.2639
    0.5727    0.5453];


%B=[0;1;0] %keeping only red

%B=[1,0, 0;0 1 0; 0 0 1];

%RGB=imread('original_panther.png');

%mask=imread('mask_panther.png');

%RRGB=imread('reflectance_panther.png');


RGB=imread('original_racoon.png');

mask=imread('mask_racoon.png');
RRGB=imread('reflectance_racoon.png');


RGB=im2double(RGB);
mask=im2double(mask);

RRGB=im2double(RRGB);

[K,J]=size(B(:,:)); %number of basis
[height,width,K]=size(RGB);

RGB(:,:,1)=RGB(:,:,1).*mask;

RGB(:,:,2)=RGB(:,:,2).*mask;

RGB(:,:,3)=RGB(:,:,3).*mask;

figure;imshow(RGB,[]);


figure;imshow(RRGB,[]);

Himg=RGB2Hue_racoon(RGB);

figure;imshow(Himg,[]);



[K,J]=size(B(:,:)); %number of basis
[height,width,K]=size(RGB);
x=zeros(3,1);
for (m=1:height) for (n=1:width)
        x(1:K,1)=RRGB(m,n,:);
        RRGB2(m,n,:)=B*B'*x;
    end
end
figure;imshow(RRGB2,[]);


%[CSHW,CSVW, CRHW,CRVW]=SetWeight(SRGB);
%[CSHW,CSVW]=SetWeight(RGB, Himg);


SRGB=RGB(60:300, 60:410,:);%nonzero border of racoon
mask=mask(60:300, 60:410,:);%nonzero border of racoon
Himg=Himg(60:300, 60:410,:);%nonzero border of racoon
%RRGB2=RRGB2(60:300, 60:410,:);%nonzero border of racoon

%SRGB=RGB(93:110,330:360,:);%small area for test for racoon
%Himg=Himg(93:110,330:360,:);%small area for test for racoon

%SRGB=SRGB(78:103,92:132,:);%small area for test for racoon
%SRGB=SRGB(60:80,180:210,:);%small area for test for panther
%Himg=Himg(78:103,92:132,:);%small area for test for racoon
%Himg=Himg(60:80,180:210,:);%small area for test for panther

%RRGB=RRGB(78:103,92:132,:);%small area for test for racoon
%RRGB=RRGB(60:80,180:210,:);%small area for test for panther

figure;imshow(SRGB,[]);
figure;imshow(Himg,[]);

%find projected reflectance image, which is the best we can recover


[K,J]=size(B(:,:)); %number of basis
[height,width,K]=size(SRGB);
x=zeros(3,1);

smin=min(min(min(SRGB)));
smax=max(max(max(SRGB)));
SRGB=(SRGB-smin)/(smax-smin);


%[CSHW,CSVW, CRHW,CRVW]=SetWeight(SRGB);
[CSHW,CSVW]=SetWeight(SRGB, Himg);

break;

lambda=2;
%ballance between constant shading and smoothness
%larger lambda gives more weighting to smoothing
lambda2=0.5;
%ballance between constant shading and smoothness and boundary condition
%arger lambda2 gives more weighting to boundary 
l1=1*lambda2;l2=1*lambda2;l3=1*lambda2;l4=1*lambda2;
%using all borders for boundary condition
%l1: right, l2: left, l3: top, l4: bottom

Q=zeros(width*height*J,width*height*J);
g1=zeros(1,K);
g2=zeros(1,K);
              
%derive weight for constant shading horizontally
SH=zeros(height*(width-1)*K, width*height*J);
for (m=1:height) 
    for (n=1:width-1)
        g1(1:K)=SRGB(m,n+1,1:K);         
        g2(1:K)=SRGB(m,n,:);
        w=CSHW(m,n);
        i=(m-1)*(width-1)*K+(n-1)*K;
        j=((n-1)*height+m-1)*J; 
        SH(i+1:i+K,j+1:j+J)=w*diag(g1)*B;
        j=((n)*height+m-1)*J;
        SH(i+1:i+K,j+1:j+J)= -w*diag(g2)*B;
    end
end
figure;imshow(SH,[]);
Q=SH'*SH;

clear SH;
DH=zeros(width*(height-1)*K, width*height*J);

%derive weight for smoothness horizontally
              
%derive weight for constant shading
for (m=1:height) 
    for (n=1:width-1)       
        w=CSHW(m,n);
        i=(m-1)*(width-1)*K+(n-1)*K;
        j=((n-1)*height+m-1)*J;
        DH(i+1:i+K,j+1:j+J)=B*(1-w);
        j=((n)*height+m-1)*J;
        DH(i+1:i+K,j+1:j+J)= -B*(1-w);
    end
end
figure;imshow(DH,[]);
Q=Q+lambda*DH'*DH;

clear DH;
SV=zeros(width*(height-1)*K, width*height*J);

%derive weight for constant shading vertically
for (n=1:width) 
    for (m=1:height-1)
               
        g1(1:K)=SRGB(m+1,n,:);         
        g2(1:K)=SRGB(m,n,:);
          
        w=CSVW(m,n);
        i=(n-1)*(height-1)*K+(m-1)*K;
        j=((n-1)*height+(m-1))*J;
        SV(i+1:i+K,j+1:j+J)=w*diag(g1)*B;
        j=((n-1)*height+(m))*J;
        SV(i+1:i+K,j+1:j+J)= -w*diag(g2)*B;
       
    end
end
figure;imshow(SV,[]);
Q=Q+SV'*SV;

clear SV;
DV=zeros(width*(height-1)*K, width*height*J);

%derive weight for smoothness vertically
for (n=1:width) 
    for (m=1:height-1)
        w=CSVW(m,n);
        
        i=(n-1)*(height-1)*K+(m-1)*K;
        j=((n-1)*height+(m-1))*J;
        DV(i+1:i+K,j+1:j+J)=B*(1-w);
        j=((n-1)*height+(m))*J;
        DV(i+1:i+K,j+1:j+J)= -B*(1-w);
        
    end
end
figure;imshow(DV,[]);

Q=Q+lambda*DV'*DV;
clear DV;

%solve for e-value and e-vec, using the evec with smallest eval as solution
%[V,D]=eig(Q);
[V,D]=eigenn(Q,width*height*J);
figure; plot(diag(D));

d(1:width*height*J',1)=diag(D);

V10=zeros(width*height*J,10);
V10=V(:,width*height*J-9:width*height*J);
save 'RacoonQmatrix.mat', 'Q', '-v7.3';
save 'RacoonEvalue.mat', 'd', '-v7.3';
save 'RacoonEvec10.mat', 'd', '-v7.3';

%the eigen values are ordered from large  to small
%following is to look at images corresponding to each eigen vectors


%Turn out two eigenvalues are much smaller than the rest and are the same. 
%using either one gives correct result but slightly wrong color  
%(if i scale each individual component to sum of individual component of the luminance image), but colors are a bit off
%If i scale
%individual components to max, majority parts get correct color except the
%top part.
%evec2 can show the correct structure only if i individually scale each
%componet to max, but most colors are off.

%trying to use the linear combination of the two eigen vectors to get the
%final result.

%I will use the sum of each coefficient over all pixels as the constraint.
%Since there are two basis, I could set up two constraints to solve for the
%coefficients a and b
%first find the coefficients of given signal

%find the corresponding reflectance image for the two smallest eigenvectors
%apply different scaling

%first find the coefficients of given signal

x=zeros(3,1);
for (m=1:height) for (n=1:width)
        x(1:K,1)=SRGB(m,n,:);
        SA(m,n,:)=B'*x;
    end
end


RR1=zeros(height,width,K);
RR2=zeros(height,width,K);


%convert coefficients to reflectance values and generate corresponding reflectance image, also generate coefficient
%images
A1=V(:,width*height*J); %stores solved coefficients in 1 column
RR1=zeros(height,width,K);
AA1=zeros(height,width,J);
x=zeros(1,K);
for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR1(m,n,:)=(B*A1(i+1:i+J))';
        x(1,1:K)=RR1(m,n,:);
        AA1(m,n,:)=x*B;
    end
end



scale=sum(sum(sum(SRGB)))/sum(sum(sum(RR1)))
RR1=scale*RR1;

figure;imshow(RR1,[]);

imwrite(RR1,'RacoonWholeRecReflectance2Basis_evec1.png');

for (k=1:K)
scale(k)=sum(sum(sum(SRGB(:,:,k))))/sum(sum(sum(RR1(:,:,k))))
RR1s(:,:,k)=scale(k)*RR1(:,:,k);
end
figure; imshow(RR1s,[])

imwrite(RR1s,'RacoonWholeRecReflectance2Basis_evec1s.png');


A2=V(:,width*height*J-1); %stores solved coefficients in 1 column
RR2=zeros(height,width,K);
AA2=zeros(height,width,J);
x=zeros(1,K);
for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR2(m,n,:)=(B*A2(i+1:i+J))';
        x(1,1:K)=RR2(m,n,:);
        AA2(m,n,:)=x*B;
    end
end


scale=sum(sum(sum(SRGB)))/sum(sum(sum(RR2)))
RR2=scale*RR2;

figure;imshow(RR2,[]);

imwrite(RR2,'RacoonWholeRecReflectance2Basis_evec2.png');

for (k=1:K)
scale(k)=sum(sum(sum(SRGB(:,:,k))))/sum(sum(sum(RR2(:,:,k))))
RR2s(:,:,k)=scale(k)*RR2(:,:,k);
end
figure; imshow(RR2s,[])

imwrite(RR2s,'RacoonWholeRecReflectance2Basis_evec2s.png');


%when i tried to combine the two reconstructed images from evecs (Coefficients) using any of the following way to
%solve a and b, none gives better results than using evec1 only. Perhaps i
%need to allow different a and b, for each color component.


%So I will use a linear combination of the two to form the final solution.
%i will try to find the linear coefficients by requiring the sum of each
%component of the evec solutions equal to the sum of the coefficients of
%each component. But the result is very similar to the result of either RR1
%or RR2.

%first find the coefficients of given signal
x=zeros(3,1);
for (m=1:height) for (n=1:width)
        x(1:K,1)=SRGB(m,n,:);
        SA(m,n,:)=B'*x;
    end
end

E11=sum(sum(AA1(:,:,1)))/(width*height);
E12=sum(sum(AA2(:,:,1)))/(width*height);

E21=sum(sum(AA1(:,:,2)))/(width*height);
E22=sum(sum(AA2(:,:,2)))/(width*height);
c1=sum(sum(SA(:,:,1)))/(width*height);
c2=sum(sum(SA(:,:,2)))/(width*height);

%equation
E=[E11, E12; E21, E22];
x=E\[c1;c2]

a=x(1); b=x(2);

AA=a*AA1+b*AA2;

x=zeros(1,J);
for (m=1:height)
    for (n=1:width)
        x(1,1:J)=AA(m,n,:);
        RR3(m,n,:)=x*B';
    end
end


figure;imshow(RR3,[]);
imwrite(RR3,'RacoonWholeRecReflectance2Basis_evec2_coefficient_sum.png');

break


%I also tried following, but none helped.
%find the coefficients by i) requiring the sum of all pixels equal to a constant (=sum of luminance values)
%ii) sum of all pixels along a particular pixel RGB value to be the same as
%in the luminance image, which looks to be under white light. I pick a
%pixel on the right side along the red line

%so let R=a R1+ b R2,
%i) gives  sum(R)=a *sum(R1) + b * sum(R2) = c1
%ii) gives
%sum(R(m0,n0))=a*sum(R1(m0,n0))+b*sum(R2(m0,n0))=c2=sum(SRGB(m0,n0)
%So i set up a 2x2 equation to solve a and b
%fix the value for some boundary points to be equal to the observed RGB value 


%set up equation for making sum of all reflectance=constant
%actually this may be better, less affected by boundary values
%Should try again if using evec solution does not work well
I=ones(1,width*height*J);
C=sum(sum(sum(SA(:,:,:))));


%fixing the right column
Y=zeros(width*height*J,1);
C1=zeros(height*J,1);
I1=zeros(height*J,width*height*J);

for (m=1:height)
    n=width;
    j=((n-1)*height+m-1)*J;
    i=(m-1)*J;
    I1(i+1:i+J,j+1:j+J)=diag([ones(1,J)]);
    C1(i+1:i+J,1)=SA(m,n,1:J);
end
Q=Q+l1*I1'*I1;
Y=Y+l1*I1'*C1;


clear I1;
clear C1;

%fixing the left column
C2=zeros(height*J,1);
I2=zeros(height*J,width*height*J);

for (m=1:height)
    n=1;
    j=((n-1)*height+m-1)*J;
    i=(m-1)*J;
    I2(i+1:i+J,j+1:j+J)=diag([ones(1,J)]);
    C2(i+1:i+J,1)=SA(m,n,1:J);
end
Q=Q+l2*I2'*I2;
Y=Y+l2*I2'*C2;


clear I2;
clear C2;
%fixing the top row 
C3=zeros(width*J,1);
I3=zeros(width*J,width*height*J);

for (n=1:width)
    m=1;
    j=((n-1)*height+m-1)*J;
    i=(n-1)*J;
    I3(i+1:i+J,j+1:j+J)=diag([ones(1,J)]);
    C3(i+1:i+J,1)=SA(m,n,1:J);
end
Q=Q+l3*I3'*I3;
Y=Y+l3*I3'*C3;


%fixing the bottom row 
C4=zeros(width*J,1);
I4=zeros(width*J,width*height*J);

for (n=1:width)
    m=height;
    j=((n-1)*height+m-1)*J;
    i=(n-1)*J;
    I4(i+1:i+J,j+1:j+J)=diag([ones(1,J)]);
    C4(i+1:i+J,1)=SA(m,n,1:J);
end
Q=Q+l4*I4'*I4;
Y=Y+l4*I4'*C4;

%recovered reflectance coefficients
R2=Q\Y;

%covert R2 to image
RR2=zeros(height,width,K);

for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR2(m,n,:)=(B*R2(i+1:i+J))';
    end
end

figure(5);imshow(RR2,[]);
%imwrite(RR2,'PantherRecReflectance2Basis.png');

%scale the recovered image
minR=min(min(RR2(:,:,1)));
maxR=max(max(RR2(:,:,1)));
minG=min(min(RR2(:,:,2)));
maxG=max(max(RR2(:,:,2)));
minB=min(min(RR2(:,:,3)));
maxB=max(max(RR2(:,:,3)));

minR,maxR,minG,maxG,minB,maxB


min1=min([minR,minG,minB]);
max1=max([maxR,maxG,maxB]);

minR=min1;minG=min1;minB=min1;
maxR=max1;maxG=max1; maxB=max1;%using this is saved as scaling 2 or s2

RR2s=zeros(height,width,K);
RR2s(:,:,1)=(RR2(:,:,1)-minR)/(maxR-minR);

RR2s(:,:,2)=(RR2(:,:,2)-minG)/(maxG-minG);
RR2s(:,:,3)=(RR2(:,:,3)-minB)/(maxB-minB);


figure;imshow(RR2s,[]);
%imwrite(RR2s,'PantherRecReflectance2Basis_s2.png');

% RecReflectance2Basis_s2.png obtained with lambda=2, lambda2=1, scale by
% common min/max

% RecReflectance2Basis_s2.png obtained with lambda=2, lambda2=1, scale by
% individual min/max

break;


%oerform eigenvalure analysis

%fix the previous Q, Y, which had extra term for I3, C3
Q=Q-l3*I3'*I3;
Y=Y-l3*I3'*C3;

[V,D]=eig(Q);
figure; plot(diag(D));

%the eigen values are ordered from small to large
%following is to look at images corresponding to each eigen vectors
for (k=1:10)
A=V(:,k); %stores solved coefficients in 1 column

%convert coefficients to reflectance values
RR=zeros(height,width,K);
for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR(m,n,:)=(B*A(i+1:i+J))';
    end
end

%find scaling factor

scale=sum(sum(sum(SRGB)))/sum(sum(sum(RR)))
RR=scale*RR;

figure;imshow(RR,[]);

end

%Turn out two eigenvalues are much smaller than the rest and are the same. 
%using either one gives correct result but slightly wrong color  
%(if i scale each individual component to sum of individual component of the luminance image), but colors are a bit off
%If i scale
%individual components to max, majority parts get correct color except the
%top part.
%evec2 can show the correct structure only if i individually scale each
%componet to max, but most colors are off.

%when i tried to combine the two evecs using any of the following way to
%solve a and b, none gives better results than using evec1 only. Perhaps i
%need to allow different a and b, for each color component.


%So I will use a linear combination of the two to form the final solution.
%I will find the coefficients by i) requiring the sum of all pixels equal to a constant (=sum of luminance values)
%ii) sum of all pixels along a particular pixel RGB value to be the same as
%in the luminance image, which looks to be under white light. I pick a
%pixel on the right side along the red line

%so let R=a R1+ b R2,
%i) gives  sum(R)=a *sum(R1) + b * sum(R2) = c1
%ii) gives
%sum(R(m0,n0))=a*sum(R1(m0,n0))+b*sum(R2(m0,n0))=c2=sum(SRGB(m0,n0)
%So i set up a 2x2 equation to solve a and b

RR1=zeros(height,width,K);
RR2=zeros(height,width,K);

A=V(:,1); %stores solved coefficients in 1 column

%convert coefficients to reflectance values
RR1=zeros(height,width,K);
for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR1(m,n,:)=(B*A(i+1:i+J))';
    end
end

for (k=1:K)
scale(k)=sum(sum(sum(SRGB(:,:,k))))/sum(sum(sum(RR1(:,:,k))))
RR1s(:,:,k)=scale(k)*RR1(:,:,k);
end

%imwrite(RR1s,'PantherRecReflectance2Basis_evec1s.png');

A=V(:,2); %stores solved coefficients in 1 column

%convert coefficients to reflectance values
RR2=zeros(height,width,K);
for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR2(m,n,:)=(B*A(i+1:i+J))';
    end
end


for (k=1:K)
scale(k)=sum(sum(sum(SRGB(:,:,k))))/sum(sum(sum(RR2(:,:,k))))
RR2s(:,:,k)=scale(k)*RR2(:,:,k);
end

%imwrite(RR2s,'PantherRecReflectance2Basis_evec2s.png');


RR1=RR1s;
RR2=RR2s;
sR1=sum(sum(sum(RR1)))/(width*height);
sR2=sum(sum(sum(RR2)))/(width*height);
c1=sum(sum(sum(SRGB)))/(width*height);
m0=30;n0=111; %pixel on the boundary that i pick
c2=sum(SRGB(m0,n0,:))
sR1p=sum(RR1(m0,n0,:));
sR2p=sum(RR2(m0,n0,:));

%equation
E=[sR1, sR2; sR1p, sR2p];
x=E\[c1;c2]

a=x(1); b=x(2);

RR=a*RR1+b*RR2;

figure;imshow(RR,[]);


%instead requiring the sum along a set of pixels (those along the right
%border, which seems to be less affected by shading) to be the same, still
%does not work. Also tried to use all borders, did not work
%matrix E is close to singular no matter what do i do

sR1p=sum(sum(RR1(:,width,:)))+sum(sum(RR1(:,1,:)))...
    + sum(sum(RR1(1,2:width-1,:)))+sum(sum(RR1(height,2:width-1,:)));

sR2p=sum(sum(RR2(:,width,:)))+sum(sum(RR2(:,1,:)))...
    + sum(sum(RR2(1,2:width-1,:)))+sum(sum(RR2(height,2:width-1,:)));
c2=sum(sum(SRGB(:,width,:)))+sum(sum(SRGB(:,1,:)))...
    + sum(sum(SRGB(1,2:width-1,:)))+sum(sum(SRGB(height,2:width-1,:)));


%equation
E=[sR1, sR2; sR1p, sR2p];
x=E\[c1;c2]

a=x(1); b=x(2);

RR=a*RR1+b*RR2;

figure;imshow(RR,[]);


%Now trying to let the boundary values to be similar to luminance  image, and solving list square problem.
%following use only the right border
%get similar result as before, but the tone is OK, still dark on the top
%when i use only right boundary

E11=0; E12=0; E22=0;
c1=0; c2=0;
for (m=1:height)
    n=1;
E11=E11+sum(RR1(m,n,:).^2);
E12=E12+sum(RR1(m,n,:).*RR2(m,n,:));
E22=E22+sum(RR2(m,n,:).^2);
c1=c1+sum(RR1(m,n,:).*SRGB(m,n,:));
c2=c2+sum(RR2(m,n,:).*SRGB(m,n,:));
end


E=[E11, E12; E12, E22];
x=E\[c1;c2]

a=x(1); b=x(2);

RR=a*RR1+b*RR2;

figure;imshow(RR,[]);


%now using all boundary result is similar to if i use only right boundary


E11=0; E12=0; E22=0;
c1=0; c2=0;
for (m=1:height)
    n=1;
E11=E11+sum(RR1(m,n,:).^2);
E12=E12+sum(RR1(m,n,:).*RR2(m,n,:));
E22=E22+sum(RR2(m,n,:).^2);
c1=c1+sum(RR1(m,n,:).*SRGB(m,n,:));
c2=c2+sum(RR2(m,n,:).*SRGB(m,n,:));
end
for (m=1:height)
    n=width;
E11=E11+sum(RR1(m,n,:).^2);
E12=E12+sum(RR1(m,n,:).*RR2(m,n,:));
E22=E22+sum(RR2(m,n,:).^2);
c1=c1+sum(RR1(m,n,:).*SRGB(m,n,:));
c2=c2+sum(RR2(m,n,:).*SRGB(m,n,:));
end
for (n=1:width)
    m=1;
E11=E11+sum(RR1(m,n,:).^2);
E12=E12+sum(RR1(m,n,:).*RR2(m,n,:));
E22=E22+sum(RR2(m,n,:).^2);
c1=c1+sum(RR1(m,n,:).*SRGB(m,n,:));
c2=c2+sum(RR2(m,n,:).*SRGB(m,n,:));
end
for (n=1:width)
    m=height;
E11=E11+sum(RR1(m,n,:).^2);
E12=E12+sum(RR1(m,n,:).*RR2(m,n,:));
E22=E22+sum(RR2(m,n,:).^2);
c1=c1+sum(RR1(m,n,:).*SRGB(m,n,:));
c2=c2+sum(RR2(m,n,:).*SRGB(m,n,:));
end


E=[E11, E12; E12, E22];
x=E\[c1;c2]

a=x(1); b=x(2);

RR=a*RR1+b*RR2;

figure;imshow(RR,[]);

RR=RR2s;
minR=min(min(RR(:,:,1)));
maxR=max(max(RR(:,:,1)));

minG=min(min(RR(:,:,2)));
maxG=max(max(RR(:,:,2)));
minB=min(min(RR(:,:,3)));
maxB=max(max(RR(:,:,3)));

minR,maxR,minG,maxG,minB,maxB

min1=min([minR,minG,minB]);

max1=max([maxR,maxG,maxB]);


%minR=min1;minG=min1;minB=min1;
%maxR=max1;maxG=max1; maxB=max1;
%scale by common min and max


RRs=zeros(height,width,K);

RRs(:,:,1)=(RR(:,:,1)-minR)/(maxR-minR);

RRs(:,:,2)=(RR(:,:,2)-minG)/(maxG-minG);
RRs(:,:,3)=(RR(:,:,3)-minB)/(maxB-minB);

figure;imshow(RRs,[]);
%pause
%end



break;

