function RGB=convRGB(specData,min_w,max_w,step)
%光谱数据卷积成RGB图像, By陈林森

    [height width specNum]=size(specData);
 
    % XYZ 积分曲线 400-720nm, 10nm
    load xyzbar.mat;
    if min_w<400 || max_w>720
        error('波长范围错误');
        exit;
    end
    
    % 插值
    x_src=400:10:720;
    r_src=xyzbar(:,1);g_src=xyzbar(:,2);b_src=xyzbar(:,3);
    
    x_dst=min_w:step:max_w;
    r_dst=spline(x_src,r_src,x_dst);
    g_dst=spline(x_src,g_src,x_dst);
    b_dst=spline(x_src,b_src,x_dst);
    
    % spectral to XYZ
    specData=reshape(specData,width*height,specNum);
    XYZ=([r_dst' g_dst' b_dst']'*specData')';
    
    XYZ=reshape(XYZ,height,width,3);
    XYZ=max(XYZ,0);
    XYZ=XYZ/max(XYZ(:));
    
    % XYZ to RGB
    RGB=XYZ2sRGB_exgamma(XYZ);
    RGB=max(RGB,0);
    RGB=min(RGB,1);
    
%     figure;imshow(RGB,'Border','tight');
    
end