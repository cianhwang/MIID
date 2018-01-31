%Recover reflectance image from a color image
%Assuming the reflectance vectors have a low rank, with basis B so that
%R_p=B A_p
%Represent all unknowns consisting of A_p for all pixels in a vector A
%Using constant shading constraint when there is significant "hue" edge so
%that SH*A=0, SV*A=0
%Using smoothness constant where there is no significant "hue" change so
%that DH*A=0, DV*A=0;
%Further letting sum of all coefficients to be the same as the sum of all coefficients of
%of input image, for each coefficient, leading to L A=C
%Solve A that minimizes L2 norm of the discrepancy from above assumptions
%J=1/2 (|SH * A|^2+ |SV*A|^2 + lambda* |DH*A|^2 +lambda |DV*A|^2 + lambda2*
% |L*A-C|^2)
%J=1/2 (A^T Q A) -  Y^ A
%where Q=SH'*SH + SV'*SV +  lambda* DH'*DH +lambda* DV'*DV +lambda2* L'*L)
%where Y=lambda2*L'*C
%Determine Q directly is faster than first getting SH etc.
%Solve using conjugate gradient method
% Yao Wang,  11/14/2015
%This version try to solve the smallest eigenvectors of matrix Q, use
%linear combination of the two based on the coefficient sum constraint
%it seems that requiring LA=C as part of the objective function gives
%somewhat better results. The method for solving two smallest evectors is
%slower than conjugat gradient

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


%RGB=im2single(RGB);
%mask=im2single(mask);

%RRGB=im2single(RRGB);
RGB=im2double(RGB);
mask=im2double(mask);
RRGB=im2double(RRGB);

%RGBsum=RGB(:,:,1)+RGB(:,:,2)+RGB(:,:,3);
%NRGB(:,:,1)=RGB(:,:,1)./(RGBsum+eps);
%NRGB(:,:,2)=RGB(:,:,2)./(RGBsum+eps);
%NRGB(:,:,3)=RGB(:,:,3)./(RGBsum+eps);
%figure;imshow(NRGB);

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

%SRGB=RGB(110:130,120:150,:);%small area for test for racoon
%Himg=Himg(110:130,120:150,:);%small area for test for racoon
%RRGB=RRGB(110:130,120:150,:);%small area for test for racoon
%mask=mask(110:130,120:150,:);%small area for test for racoon



%SRGB=RGB(60:149,260:350,:);%small area for test for racoon
%Himg=Himg(60:149,260:350,:);%small area for test for racoon
%RRGB=RRGB(60:149,260:350,:);%small area for test for racoon
%mask=mask(60:149,260:350,:);%small area for test for racoon

%medium large region (a good area to test)
%SRGB=RGB(110:170,120:190,:);%small area for test for racoon
%SRGB=SRGB(60:80,180:210,:);%small area for test for panther
%Himg=Himg(110:170,120:190,:);%small area for test for racoon
%Himg=Himg(60:80,180:210,:);%small area for test for panther
%RRGB=RRGB(110:170,120:190,:);%small area for test for racoon
%mask=mask(110:170,120:190,:);%small area for test for racoon

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
DRHW=1-CSHW;
DRVW=1-CSVW;
CSVW=CSVW.*CSVW;
CSHW=CSHW.*CSHW;
DRHW=DRHW.*DRHW;
DRVW=DRVW.*DRVW;


%break;

lambda=2;
%ballance between constant shading and smoothness
%larger lambda gives more weighting to smoothing
lambda2=0.5;
%ballance between constant shading and smoothness and boundary condition
%arger lambda2 gives more weighting to boundary 
TCS=0.0001; %weight threshold. If CSW<T, will not force the constant shading constraint
TDR=0.0001;%%weight threshold. If DRW<T, will not force the constant reflectance constraint
%not used, all pixels are considered.

g1=zeros(1,K);
g2=zeros(1,K);
              
N=width*height*J;
diagI=[1, 0; 0, 1];
Q=zeros(N,N);
%derive weight for constant shading horizontally
QSH=zeros(N,N);
QDH=zeros(N,N);
for (m=1:height) 
    for (n=2:width-1)
        g1(1:K)=SRGB(m,n,1:K);         
        g2(1:K)=SRGB(m,n+1,:);       
        g0(1:K)=SRGB(m,n-1,:);
        w12=CSHW(m,n); w01=CSHW(m,n-1);
        d12=DRHW(m,n); d01=DRHW(m,n-1);
       
           
        i=((n-1)*height+m-1)*J;
        j=((n-1)*height+m-1)*J;
        QSH(i+1:i+J,j+1:j+J)=w12*B'*diag(g2.*g2)*B+w01*B'*diag(g0.*g0)*B;
        QDH(i+1:i+J,j+1:j+J)=(d12+d01)*diagI;
    
        j=((n-2)*height+m-1)*J;
        QSH(i+1:i+J,j+1:j+J)= -w01*B'*diag(g0.*g1)*B;
        QDH(i+1:i+J,j+1:j+J)= -d01*diagI;
        j=((n)*height+m-1)*J;
        
        QSH(i+1:i+J,j+1:j+J)= -w12*B'*diag(g2.*g1)*B;
        
        QDH(i+1:i+J,j+1:j+J)= -d12*diagI;
       
    end
        
    n=1;
        g1(1:K)=SRGB(m,n,1:K);         
        g2(1:K)=SRGB(m,n+1,:);       
        %g0(1:K)=SRGB(m,n-1,:);
        w12=CSHW(m,n); %w10=CSHW(m,n-1);
        d12=DRHW(m,n); %w10=CSHW(m,n-1);
        i=((n-1)*height+m-1)*J;
        j=((n-1)*height+m-1)*J;
        QSH(i+1:i+J,j+1:j+J)=w12*B'*diag(g2.*g2)*B;
        
        QDH(i+1:i+J,j+1:j+J)=d12*diagI;
        j=((n)*height+m-1)*J;
        QSH(i+1:i+J,j+1:j+J)= -w12*B'*diag(g2.*g1)*B;
        
        QDH(i+1:i+J,j+1:j+J)=-d12*diagI;
        
        
    n=width;
        g1(1:K)=SRGB(m,n,1:K);         
        %g2(1:K)=SRGB(m,n+1,:);       
        g0(1:K)=SRGB(m,n-1,:);
        %w12=CSHW(m,n); 
        w10=CSHW(m,n-1);
        %d12=DRHW(m,n); 
        d10=DRHW(m,n-1);
        
        i=((n-1)*height+m-1)*J;
        j=((n-1)*height+m-1)*J;
        QSH(i+1:i+J,j+1:j+J)=w01*B'*diag(g0.*g0)*B;
        
        QDH(i+1:i+J,j+1:j+J)=d01*diagI;
        j=((n-2)*height+m-1)*J;
        QSH(i+1:i+J,j+1:j+J)= -w01*B'*diag(g0.*g1)*B;
        
        QDH(i+1:i+J,j+1:j+J)=-d01*diagI;
        
end

Q=Q+QSH;
Q=Q+lambda*QDH; 
%figure;imshow(QSH,[]);
%figure;imshow(QDH,[]);
clear QSH;
clear QDH;
QSV=zeros(N,N);
QDV=zeros(N,N);

for (n=1:width)
    for (m=2:height-1) 
        g1(1:K)=SRGB(m,n,1:K);         
        g2(1:K)=SRGB(m+1,n,:);       
        g0(1:K)=SRGB(m-1,n,:);
        w12=CSVW(m,n); w01=CSVW(m-1,n);
        d12=DRVW(m,n); d01=DRVW(m-1,n);
       
           
        i=((n-1)*height+m-1)*J;
        j=((n-1)*height+m-1)*J;
        QSV(i+1:i+J,j+1:j+J)=w12*B'*diag(g2.*g2)*B+w01*B'*diag(g0.*g0)*B;
        QDV(i+1:i+J,j+1:j+J)=(d12+d01)*diagI;
    
        j=((n-1)*height+m-2)*J;
        QSV(i+1:i+J,j+1:j+J)= -w01*B'*diag(g0.*g1)*B;
        QDV(i+1:i+J,j+1:j+J)= -d01*diagI;
        j=((n-1)*height+m)*J;
        
        QSV(i+1:i+J,j+1:j+J)= -w12*B'*diag(g2.*g1)*B;
        
        QDV(i+1:i+J,j+1:j+J)= -d12*diagI;
       
    end
        
    m=1;
        g1(1:K)=SRGB(m,n,1:K);         
        g2(1:K)=SRGB(m+1,n,:);       
        %g0(1:K)=SRGB(m,n-1,:);
        w12=CSVW(m,n); %w10=CSHW(m,n-1);
        d12=DRVW(m,n); %w10=CSHW(m,n-1);
        i=((n-1)*height+m-1)*J;
        j=((n-1)*height+m-1)*J;
        QSV(i+1:i+J,j+1:j+J)=w12*B'*diag(g2.*g2)*B;
        
        QDV(i+1:i+J,j+1:j+J)=d12*diagI;
        j=((n-1)*height+m)*J;
        QSV(i+1:i+J,j+1:j+J)= -w12*B'*diag(g2.*g1)*B;
        
        QDV(i+1:i+J,j+1:j+J)=-d12*diagI;
        
    m=height;
        g1(1:K)=SRGB(m,n,1:K);         
        %g2(1:K)=SRGB(m+1,n,:);       
        g0(1:K)=SRGB(m-1,n,:);
        w01=CSVW(m-1,n); %w10=CSHW(m,n-1);
        d01=DRVW(m-1,n); %w10=CSHW(m,n-1);
        i=((n-1)*height+m-1)*J;
        j=((n-1)*height+m-1)*J;
        QSV(i+1:i+J,j+1:j+J)=w01*B'*diag(g0.*g0)*B;
        
        QDV(i+1:i+J,j+1:j+J)=d01*diagI;
        j=((n-1)*height+m-2)*J;
        QSV(i+1:i+J,j+1:j+J)= -w01*B'*diag(g0.*g1)*B;
        
        QDV(i+1:i+J,j+1:j+J)=-d01*diagI;
        
end

Q=Q+QSV;
Q=Q+lambda*QDV; 
%figure;imshow(QSV,[]);
%figure;imshow(QDV,[]);
%figure;imshow(Q,[]);

clear QSV;
clear QDV;

disp('Complete computing Q!');

%following programs find smalles evalues and associated e-vectors, smallest
%firt
OPTS.SIGMA='SE';
OPTS.K=2;
OPTS.NBLS=5;
[V, D, PRGINF]=irbleigs(Q,OPTS);
disp('eigen values')
diag(D)

%break;



RR1=zeros(height,width,K);
RR2=zeros(height,width,K);


%convert coefficients to reflectance values and generate corresponding reflectance image, also generate coefficient
%images
A1=V(:,1); %stores solved coefficients in 1 column
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

figure;imshow(RR1,[0 1]);

imwrite(RR1,'RacoonWholeRecReflectance_evec1.png');



A2=V(:,2); %stores solved coefficients in 1 column
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

figure;imshow(RR2,[0 1]);

imwrite(RR2,'RacoonWholeRecReflectance_evec2.png');

%now set up the equations to solve coefficients for combining two evectors

%only consider the masked area
%use the following constraint on the sum of each A component of the
%normalized NGB image (chroma only).


SA=zeros(height, width, J);
x=zeros(3,1);
for (m=1:height) for (n=1:width)
        x(1:K,1)=SRGB(m,n,:);
        SA(m,n,:)=B'*x;
    end
end


E11=sum(sum(AA1(:,:,1).*mask(:,:)))/(width*height);
E12=sum(sum(AA2(:,:,1).*mask(:,:)))/(width*height);

E21=sum(sum(AA1(:,:,2).*mask(:,:)))/(width*height);
E22=sum(sum(AA2(:,:,2).*mask(:,:)))/(width*height);
c1=sum(sum(SA(:,:,1).*mask(:,:)))/(width*height);
c2=sum(sum(SA(:,:,2).*mask(:,:)))/(width*height);

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


figure;imshow(RR3,[0 1]);
imwrite(RR3,'RacoonWholeRecReflectance_evec12.png');

figure; imshow(RRGB, [0 1]);

break;

SA=zeros(height, width, J);
x=zeros(3,1);
for (m=1:height) 
    for (n=1:width)
        x(1:K,1)=SRGB(m,n,:).*mask(m,n);
        SA(m,n,:)=B'*x;
    end
end

L=zeros(2,width*height*J);
for (m=1:height)
    for (n=1:width)
    L(1,((n-1)*height+m-1)*J+1)=mask(m,n);
    L(2,((n-1)*height+m-1)*J+2)=mask(m,n);
    end
end
C=zeros(2,1);
C(1)=sum(sum(SA(:,:,1)));
C(2)=sum(sum(SA(:,:,2)));

OPTS.SIGMA='SE';
OPTS.K=3;
[V,D]=irbleigs(Q,OPTS);

break;

Q2=Q+lambda2*L'*L;
Y2=lambda2*L'*C;

%solve directly
%A1=Q\Y2;

%use conjgate gradient method
TOL=1e-6;
[A1,flag,relres,iter]=pcg(Q2,Y2,TOL,200);
flag
relres
iter


%convert coefficients to reflectance values and generate corresponding reflectance image, also generate coefficient
%images

x=zeros(1,K);
for (m=1:height)
    for (n=1:width)
        i=((n-1)*height+m-1)*J;
        RR(m,n,:)=(B*A1(i+1:i+J))';
    end
end



figure;imshow(RR,[0 1]);
figure;imshow(RRGB,[0 1]);

RR(:,:,1)=RR(:,:,1).*mask;
RR(:,:,2)=RR(:,:,2).*mask;
RR(:,:,3)=RR(:,:,3).*mask;
figure;imshow(RR);


break;

imwrite(RR,'RacoonWholeRecRefCG2.png');

break

%when using the total sum, the matrix is ill conditioned when lambda2 is
%small

%following for sum of R, not used
LB=ones(1,K)*B;
l0=0;
L=zeros(1,width*height*J);
for (n=1:width*height)
    L(1,l0+1:l0+J)=LB;
end
C=[sum(sum(sum(RRGB)))];

%following for sum of A

SA=zeros(height, width, J);
x=zeros(3,1);
for (m=1:height) for (n=1:width)
        x(1:K,1)=RRGB(m,n,:);
        SA(m,n,:)=B'*x;
    end
end

%use the following constraint on the sum of each A component
l0=0;
L=zeros(2,width*height*J);
for (n=1:width*height)
    L(1,l0+1)=1;
    L(2,l0+2)=1;
    l0=l0+2;
end
C=zeros(2,1);
C(1)=sum(sum(SA(:,:,1)));
C(2)=sum(sum(SA(:,:,2)));

