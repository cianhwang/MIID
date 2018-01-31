

%set weight based on input luminance color image L
%CSHW, CSVW is weight for enforcing constant shading, larger when there is significant change in color
%CSHW depends on horizontal difference
%CSVW depends on vertical difference
%CRHW, CRHW is weight for encoding constant reflectance, larger when there is small
%changes in color
%I use CRHW=1-CSHW here

%Yao Wang 2015/11/8
%function [CSHW,CSVW, CRHW, CRVW]=SetWeight(L, H)
%L is the RGB image, H is the Hue image

function [CSHW,CSVW]=SetWeight(L, H)

[height,width,K]=size(L);


a=0.05;%threshold for RGB vector difference
b=0.5;
alpha=50;
beta=2;
T0=0.00001;
[height,width,K]=size(L);
g1=zeros(K,1);g2=zeros(K,1);
CSHW=zeros(height,width);
CSVW=zeros(height,width);
%CRHW=zeros(height,width);
%CRVW=zeros(height,width);


for (m=1:height)
    for (n=1:width-1)
        g1(:,1)=L(m,n,:);
        g2(:,1)=L(m,n+1,:);
        g1n=sum(g1.^2);
        g2n=sum(g2.^2);
        if (g1n>=T0 || g2n>=T0)
            %g1=g1/sqrt(g1n);
            %g2=g2/sqrt(g2n);
            %c=sum((g1(1:3)-g2(1:3)).^2);%better than using CbCr difference
            %c=g1(2)*g2(2)+g1(3)*g2(3);%inner product of CbCr
            %This is cos of the angle between g1 and g2 
            %w small if g1 and g2 have very different color
            %should enforce the ratio, relax smoothness
            %alternatively using Gaussian?
            %c=sum((g1(2:3)-g2(2:3)).^2)/2;
            %sigma1=(std([L(m,n,1),L(m,n-1,1),L(m,n-2,1)])+...
            %    std([L(m,n,2),L(m,n-1,2),L(m,n-2,2)])+...
            %    std([L(m,n,3),L(m,n-1,3),L(m,n-2,3)]))/3;
            %sigma2=(std([L(m,n,1),L(m,n+1,1),L(m,n+2,1)])+...
            %    std([L(m,n,2),L(m,n+1,2),L(m,n+2,2)])+...
            %    std([L(m,n,3),L(m,n+1,3),L(m,n+2,3)]))/3;
            %sigma=(std([L(m,n,1),L(m,n-1,1),L(m,n-2,1),L(m,n+1,1),L(m,n+2,1)])+...
            %    std([L(m,n,2),L(m,n-1,2),L(m,n-2,2),L(m,n+1,2),L(m,n+2,2)])+...
            %    std([L(m,n,3),L(m,n-1,3),L(m,n-2,3),L(m,n+1,3),L(m,n+2,3)]))/3;
            %s=(sigma1+sigma2)/sigma/2;
            %c=sum(abs(H(m-1:m+1,n)-H(m-1:m+1,n+1)));
            c=abs(H(m,n)-H(m,n+1));
            %c=abs((H(m-1,n)-H(m-1,n+1))+2*(H(m,n)-H(m,n+1))+(H(m+1,n)-H(m+1,n+1)))/3;
            wc=1/(1+exp(-alpha*(c-a)));
             
            
            %ws=1/(1+exp(beta*(s-b)));
            w=wc;
            
            %CHimg(m,n)=c;
            %Simg(m,n)=s;
            %c,s,wc, ws, w,
          
            CSHW(m,n)=w;
            
        end
    end
end

for (m=1:height-1)
    for (n=1:width)
        g1(:,1)=L(m,n,:);
        g2(:,1)=L(m+1,n,:);
        g1n=sum(g1.^2);
        g2n=sum(g2.^2);
        g1n=g1(1); %Y value
        g2n=g2(1);
        if (g1n>=T0 || g2n>=T0)
            %g1=g1/sqrt(g1n);
            %g2=g2/sqrt(g2n);
            %c=sum((g1(1:3)-g2(1:3)).^2);%better than using CbCr difference
            %c=g1(2)*g2(2)+g1(3)*g2(3);%inner product of CbCr
            %This is cos of the angle between g1 and g2 
            %w small if g1 and g2 have very different color
            %should enforce the ratio, relax smoothness
            %alternatively using Gaussian?
            %c=sum((g1(2:3)-g2(2:3)).^2)/2;
            %sigma1=(std([L(m,n,1),L(m,n-1,1),L(m,n-2,1)])+...
            %    std([L(m,n,2),L(m,n-1,2),L(m,n-2,2)])+...
            %    std([L(m,n,3),L(m,n-1,3),L(m,n-2,3)]))/3;
            %sigma2=(std([L(m,n,1),L(m,n+1,1),L(m,n+2,1)])+...
            %    std([L(m,n,2),L(m,n+1,2),L(m,n+2,2)])+...
            %    std([L(m,n,3),L(m,n+1,3),L(m,n+2,3)]))/3;
            %sigma=(std([L(m,n,1),L(m,n-1,1),L(m,n-2,1),L(m,n+1,1),L(m,n+2,1)])+...
            %    std([L(m,n,2),L(m,n-1,2),L(m,n-2,2),L(m,n+1,2),L(m,n+2,2)])+...
            %    std([L(m,n,3),L(m,n-1,3),L(m,n-2,3),L(m,n+1,3),L(m,n+2,3)]))/3;
            %s=(sigma1+sigma2)/sigma/2;
            %c=sum(abs(H(m,n-1:n+1)-H(m+1,n-1:n+1)));
              c=abs(H(m,n)-H(m+1,n));
              %c=abs((H(m,n-1)-H(m+1,n-1))+2*(H(m,n)-H(m+1,n))+(H(m,n+1)-H(m+1,n+1)))/3;
           
            wc=1/(1+exp(-alpha*(c-a)));
            
            %ws=1/(1+exp(beta*(s-b)));
             
            %if c>=a 
            %    wc=1
            %else
            %    wc=0
            %end
            w=wc;
            
            %CVimg(m,n)=c;
            %Simg(m,n)=s;
            %c,s,wc, ws, w,
          
            CSVW(m,n)=w;
               
        end
    end
end

figure; imshow(CSHW,[]);

figure; imshow(CSVW,[]);

%CRHW=1-CSHW;
%CRVW=1-CSVW;


%figure; imshow(CRHW,[]);

%figure; imshow(CRVW,[]);
