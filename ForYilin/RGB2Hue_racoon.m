function [H] = RGB2Hue(L)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

r=L(:,:,1);
g=L(:,:,2);
b=L(:,:,3);

S=1-3.*(min(min(r,g),b))./(r+g+b+eps);
I=(r+g+b)/3;
th=acos((0.5*((r-g)+(r-b)))./((sqrt((r-g).^2+(r-b).*(g-b)))+eps));
%this can get very noisy results for color that is white
%or for dark pixels which are very similar to purple, which has high hue,
%but low intensity, not sure what is the problem with this formula
H=th;
H(b>g)=2*pi-H(b>g);
H=H/(2*pi);
S=1-3.*(min(min(r,g),b))./(r+g+b+eps);
I=(r+g+b)/3;
figure; imshow(H,[]);

imshow(H,[]);
%H=medfilt2(H,[3 1]);
%H=medfilt2(H,[1 3]);

imshow(H,[]);

%none of the following i tried to get rid of the noisy white spot in gray
%area worked. So i finally decided to just do median filtering.
%those noisy points are very similar to purple colored points. So some
%isolated purple points will be removed unfortunately.
%pixels in gray area can get very high hue value, purely due to random
%varitation, but so is purple
%purple pixels have very low saturation and high hue
%trying to identify area which are gray but with inconsistent hue
%figure;imshow(S,[]);
%figure,imshow(I,[]);
%Egr=(g-r)./(g+r+eps);
%Egb=(g-b)./(g+b+eps);
%Erb=(r-b)./(r+b+eps);
%E=((abs(Egr)<0.05) & (abs(Egb)<0.05)&(abs(Erb)<0.05) & ((r+g+b)>0));

%for panther
%E=(((S<0.1) &(((H<0.39) & (H>0.05))) & (I>0.15)));
%figure; imshow(E)
%E=(E|((S<0.05) & (H>0.8)& (I<0.05))); 
%figure; imshow(E)

%The following worked well for racoon (actually same as panther

%E=((S<0.1) &(( ((H<0.39) & (H>0.05))) & (I>0.20))); 
E=((S<0.1) &(( ((H<0.39) & (H>0.05))) & (I>0.15))); 
E=(E|((S<0.05) & (H>0.8)& (I<0.05)));
figure; imshow(E)

%E=((S<0.1) &((H>0.9) | ((H<0.39) & (H>0.05))) & (I>0.20)); figure; imshow(E)
%green pixels can have H slighty lower than 0.4, and intensity around 0.2 
% the current threshold is a ballance
%pixels in these range are gray pixels which due to slight variation
%creates edges in hue image, want to reset them all to white color
%E=((S<0.01)& (H>0.8));figure;imshow(E); 
%when Hue is low, saturation is usually high for a red color
%when saturation is low and hue is low, it is noise
E=1-E;
E=bwareaopen(E,5);%to remove noise in the mask, due to dark pixels which look like purple.
E=1-E;
%E=imopen(E,1);
white=acos(0)/2/pi; %Hue corresponding to r=g=b
H=H.*(1-E)+E*white;



E=(I>0);
H=H.*E+(1-E)*(-0.25);
%so that masked pixels get a hue = -0.25

figure;imshow(H);


end

