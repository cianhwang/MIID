clc;clear;close all

nameList = {'apple'; 'box'; 'cup1'; 'cup2'; 'deer'; 'dinosaur'; 'frog1'; 'frog2';...
    'panther'; 'paper1'; 'paper2'; 'pear'; 'phone'; 'potato'; 'raccoon'; 'squirrel'; 'sun'; ...
    'teabag1'; 'teabag2'; 'turtle'};
%-----------------------------0122
gpu = gpuDevice(1);
shadmat = zeros(10,10, 20);
refmat = zeros(10,10, 20);
%
for name =1%:length(nameList)
FILENAME = char(nameList(name));
Name = strcat('.\MITdata\',  FILENAME);
addpath(Name);
%% read spectral data
% [specImg,startw,endw,stepw]=readHSD('E:\GroundTruth\plaster\diffuse.hsd');
% load('diffuse');
% load('reflectance');
% load('shading');
% gt_shad = shading; 
% specImg = diffuse; 
% gt_refl =  ref;%lectance;
% 
% 
% mask = imread('mask.bmp');
% mask = mean(mask, 3);
% mask = logical(mask);
% groundtruth;

gt_shad = imread('shading.png'); gt_shad = im2double(gt_shad);
specImg =  imread('diffuse.png'); specImg = im2double(specImg);
gt_refl =   imread('reflectance.png'); gt_refl = im2double(gt_refl);


mask = imread('mask.png');
mask = mean(mask, 3);
mask = logical(mask);




%---------------block--------------

[block1, block2] = find(mask == 1);
block1 = min(block1):max(block1);
block2 = min(block2):max(block2);

specImg = specImg(block1, block2, :);
mask = mask(block1, block2, :);
gt_shad = gt_shad(block1, block2, :);
gt_refl = gt_refl(block1, block2, :);
% 
% SIZE = 0.3;
% specImg = imresize(specImg, SIZE);
% mask = imresize(mask, SIZE);
% gt_shad = imresize(gt_shad, SIZE);
% gt_refl = imresize(gt_refl, SIZE);
%  
% block1 = 176:215;
% block2 = 146:185;
% 
% % block1 = 61:338;
% % block2 = 170:735;
% specImg = specImg(block1, block2, :);
% mask = mask(block1, block2, :);
% gt_shad = gt_shad(block1, block2, :);
% gt_refl = gt_refl(block1, block2, :);

smin=min(specImg(:));
smax=max(specImg(:));
specImg=(specImg-smin)/(smax-smin);

[Row, Col, Chn] = size(specImg);
K = Chn;
N = Row*Col;

for k = 1:K
    specImg(:, :, k) = specImg(:, :, k).*mask;
end
specImgVec = reshape(specImg, [], K);

figure, imshow(specImg);
%% Bs, Br
%-----------Br---------------
% Jr = 2;
% % load('munsell400_700_5');
% % coeff = pca(munsell(1:K, :)');
% coeff = pca(specImgVec);
% Br = coeff(:, 1:Jr);

Jr = 3; Br = eye(3);

%-----------Bs---------------
% Js = 1;
% load('redhead_illumination');
% Bs = illumination';
% Bs = Bs/norm(Bs);

% shadVec = reshape(shading, [], 59);
% coeff = pca(shadVec);
% Js = 2; Bs = coeff(1:2:59, 1:Js);

Js = 1; Bs = ones(K, 1)/(K^0.5);

%% Weightmap
beta = 5e-4;
H = Normalized2(specImg);
[Wr, Wd] = weightMap(H, beta, 0);
WrWr = reshape(Wr, [],1);
WdWd = reshape(Wd, [],1);
HVec = reshape(H, [], K);

%% E s-term

    %---------------------sparseBs-----------------------

    Bs_temp = sparse(Bs);
    Bs_temp = repmat({Bs_temp}, N, 1);
    sparseBs = blkdiag(Bs_temp{:});

    NeighSize = 4;
    rowIndex = zeros(1,NeighSize*N*K*2); rowIndex(1:2) = [1,1];
    colIndex = zeros(1,NeighSize*N*K*2);
    
    WscVal = zeros(1,NeighSize*N*K*2);
    Wsc_LVal = zeros(1,NeighSize*N*K*2);
    Vrc_LVal = zeros(1,NeighSize*N*K*2);
    VrcVal = zeros(1,NeighSize*N*K*2);

    idx = 1;        
    for Neighbor=1:NeighSize
    for i = 1:1:N % ÐÐ
        coor = [i+1,i+Row,i-1,i-Row];
        p = i;
            if (coor(Neighbor)<N) && (coor(Neighbor)>0)
                q = coor(Neighbor);
                % compute w v
                if Neighbor ==1
                    w = Wd(p);
                elseif Neighbor==2
                    w = Wr(p);
                elseif Neighbor==3
                    w = Wd(q);
                else
                    w = Wr(q);
                end                
                v = 1-w;
                
                for k = 1:K
                    if idx > 1
                    rowIndex(idx:idx+1) = [rowIndex(idx-1)+1, rowIndex(idx-1)+1];
                    end
                    colIndex(idx:idx+1) = [K*(p-1)+k, K*(q-1)+k];
                    
                    WscVal(idx:idx+1) = [w, -w];
                    Wsc_LVal(idx:idx+1) = [w*specImgVec(q, k), -w*specImgVec(p, k)];
                    Vrc_LVal(idx:idx+1) = [v*specImgVec(q, k), -v*specImgVec(p, k)];
                    VrcVal(idx:idx+1) = [v, -v];
                    
                    idx = idx+2;
                end
            end
        end
    end
    zeroMark = find(rowIndex == 0);
    rowIndex(zeroMark) = []; colIndex(zeroMark) = [];
    
    WscVal(zeroMark) = [];    VrcVal(zeroMark) = [];	Wsc_LVal(zeroMark) = [];    Vrc_LVal(zeroMark) = [];
    
    Wsc = sparse(rowIndex, colIndex, WscVal, N*NeighSize*K, N*K)*sparseBs;
    Vrc_L = sparse(rowIndex, colIndex, Vrc_LVal, N*NeighSize*K, N*K)*sparseBs;
    Wsc = gpuArray(Wsc);
    Vrc_L = gpuArray(Vrc_L);
    
    % Generic_s
    %---------------------------
    L = speye(N*K);     L = L*sparseBs;
    C = reshape(specImgVec', [], 1);
    L= gpuArray(L);
    C = gpuArray(C);
    %-----------------------------
    
%         L=zeros(K,N*K);
%         C=zeros(K,1);
%         for j = 1:N
%             for k = 1:K
%                 L(k,(j-1)*K+k)=mask(j);
%                 C(k) =  C(k) + specImgVec(j, k);
%             end
%         end
%         L = sparse(L)*sparseBs;
    
%% E r-term

    %---------------------sparseBs-----------------------


    Br_temp = sparse(Br);
    Br_temp = repmat({Br_temp}, N, 1);
    sparseBr = blkdiag(Br_temp{:});

    %% R estimate
    Wsc_L = sparse(rowIndex, colIndex, Wsc_LVal, N*NeighSize*K, N*K)*sparseBr;
    Vrc = sparse(rowIndex, colIndex, VrcVal, N*NeighSize*K, N*K)*sparseBr;
    Wsc_L = gpuArray(Wsc_L);
    Vrc = gpuArray(Vrc);
	% Generic_r
    %---------------------------
    L_r = speye(N*K); L_r = L_r*sparseBr;
    C_r = reshape(specImgVec', [], 1);
    L_r = gpuArray(L_r);
    C_r = gpuArray(C_r);
    %----------------------------------
%     L_r=zeros(1,N*K);
%     C_r=zeros(1,1);
%     for j = 1:2:N
%         for k = 1
%             L_r(k,(j-1)*K+k)=mask(j);
%             C_r(k) =  C_r(k) + specImgVec(j, k);
%         end
%     end
%     L_r = sparse(L_r)*sparseBr;

%% Global
%     SUPER_PIXEL_KMEANS_NUM = 20;
%     vSupPixImg = HVec;
%     vSupPixImgL2Norm = GetL2Norm(vSupPixImg);
%     vSupPixClusterMap = kmeans(vSupPixImgL2Norm,SUPER_PIXEL_KMEANS_NUM,'distance','cosine');
%     Map = reshape(vSupPixClusterMap, [Row, Col]);
%     figure, imshow(Map, [])
%     colormap('default');
% 
%     GrowIndex = zeros(1,N*K*2);  GrowIndex(1:2) = [1,1];
%     GcolIndex = zeros(1,N*K*2);
%     GVrcVal = zeros(1,N*K*2);
%     idx=1;
%     
%     for i = 1:SUPER_PIXEL_KMEANS_NUM
%         sClusterIdxs = find(vSupPixClusterMap == i);
%         sClusterSize = length(sClusterIdxs);
%         if sClusterSize == 1
%             continue;
%         end
%         for j = 1:sClusterSize
%             p = sClusterIdxs(j);
%             while 1
%                 sRandiRes = randi(sClusterSize,1);
%                 if sRandiRes ~= j
%                     break;
%                 end
%             end
%             q = sClusterIdxs(sRandiRes);
%             for k = 1:K
%                 if idx > 1
%                     GrowIndex(idx:idx+1) = [GrowIndex(idx-1)+1, GrowIndex(idx-1)+1];
%                 end
%                 GcolIndex(idx:idx+1) = [K*(p-1)+k, K*(q-1)+k];
%                 GVrcVal(idx:idx+1) = [specImgVec(q, k), -specImgVec(p, k)];
%                 
%                 idx = idx+2;
%             end
%         end
%     end
%     zeroMark = find(GrowIndex == 0);
%     GrowIndex(zeroMark) = [];    GcolIndex(zeroMark) = [];    GVrcVal(zeroMark) = [];
%     Gc=sparse(GrowIndex,GcolIndex,GVrcVal,N*K,N*K)*sparseBs;


%% Estimate s
lambda_sc = 1;%00000;
lambda_gc = 0;%.1;
%-----------------------------------

idxI = 0;
idxJ = 0;
for lambda_rc = logspace(1, 3, 10)%00000;%2.559547922699536e+02;
    idxI = idxI+1;
    %% S estimate

    for lambda_esti_s = logspace(-4, -1, 10)%0.0005;%0.01;%0.00138949549437300;
%     Wsc = sparse(rowIndex, colIndex, WscVal, N*NeighSize*K, N*K)*sparseBs;
%     Vrc_L = sparse(rowIndex, colIndex, Vrc_LVal, N*NeighSize*K, N*K)*sparseBs;
%     Gcs = Gc*sparseBs;
    idxJ = idxJ +1;

        
    b = lambda_esti_s*L'*C;
    Q = lambda_sc*(Wsc'*Wsc) + lambda_rc*(Vrc_L'*Vrc_L) +lambda_esti_s*(L'*L);%+ lambda_gc*(Gc'*Gc);
    disp('Q s-estimate complete!')
    
    %--------------------------------------------------
%     save 'para.mat' Js N Bs Row Col K mask
    tic,
    TOL=1e-8;
    [shadVec,flag,relres,iter]=pcg(Q,b,TOL,2000);
    flag
    relres
    iter
    toc
    %---------------------------------------------------
%     specImgLongVector =specImgVec'; specImgLongVector = sparseBs'*specImgLongVector(:);
%     lambda_max = norm(cat(1,Wsc,2*Vrc_L)'*cat(1,-Wsc*specImgLongVector,-2*Vrc_L*specImgLongVector), 'inf');
%     lambda = 0.1*lambda_max;
% 	[shadVec history] = lasso(cat(1,Wsc,2*Vrc_L), cat(1,-Wsc*specImgLongVector,-2*Vrc_L*specImgLongVector),lambda_esti_s, 1.0, 1.0);
%     shadVec = specImgLongVector+shadVec;
%     [shadVec history] = lad(Q, b, 1, 1.0);
%     [shadVec history] = lasso(Wsc+Vrc_L, -Wsc*specImgLongVector-Vrc_L*specImgLongVector, lambda, 1.0, 1.0);
    
    a=reshape(shadVec,[Js,N]);
    r=(Bs*a)';
    
    shadImg = zeros(Row, Col, K);
    for k = 1:K
        shadImg(:, :, k) = reshape(gather(r(:, k)), Row, Col).*mask;
    end
    shadImg = (shadImg-min(shadImg(:)))/(max(shadImg(:))-min(shadImg(:)));
    figure, imshow(shadImg, [])
    derived_reflImg = specImg./(shadImg+eps);
    derived_reflImg = (derived_reflImg-min(derived_reflImg(:)))/(max(derived_reflImg(:))-min(derived_reflImg(:)));
    figure, imshow(derived_reflImg, [])
    score = LMSE( mean(gt_shad, 3), mean(shadImg,3), mask)
    score2 = LMSE( mean(gt_refl, 3), mean(derived_reflImg,3), mask)
    imwrite(shadImg, strcat(FILENAME, '_shading', num2str(score),'.png'));
     imwrite(derived_reflImg, strcat(FILENAME, '_derrefl', num2str(score2),'.png'));




    %--------------------------------------------0122---------------------------------
    lambda_esti_r = lambda_esti_s;%0.0005;%0.005;
    lambda_data = 1;
    
    shadVec_restore = sparseBs*shadVec;
    shadVec_restore = gather(shadVec_restore);
    Qdata = diag(sparse(shadVec_restore)) * sparseBr;
    Qdata = gpuArray(Qdata);
    Cdata = reshape(specImgVec', [], 1);

    b = lambda_esti_r*L_r'*C_r;
    Q = lambda_sc*(Wsc_L'*Wsc_L) + lambda_rc*(Vrc'*Vrc) +lambda_esti_r*(L_r'*L_r);    
    Q = Q + lambda_data*(Qdata'*Qdata);
    b = b + lambda_data*Qdata'*Cdata;
    disp('Q r-estimate complete!')

    tic,
    TOL=1e-8;
    [reflVec,flag,relres,iter]=pcg(Q,b,TOL,1000);
    flag
    relres
    iter
    toc

    %  l1 form is too complex... para
    %     Q = cat(1,lambda_sc*Wsc_L, lambda_rc*Vrc); 
    %     L_r = cat(1, sparseBr, lambda_data*Qdata);
    %     C = cat(1, C, lambda_data*Cdata);
    %     b = -Q*L_r'*C;
    %     
    % %     specImgLongVector =specImgVec'; specImgLongVector = sparseBr'*specImgLongVector(:);
    %     lambda_max = norm(Q*b, 'inf');
    %     lambda = 0.1*lambda_max;    
    % 	[reflVec history] = lasso(Q, b, lambda_esti_r, 1.0, 1.0);
    %     reflVec = specImgLongVector+reflVec;
    
    a=reshape(reflVec,[Jr,N]);
    r=(Br*a)';
    
    reflImage = zeros(Row, Col, K);
    for k = 1:K
        reflImage(:, :, k) = reshape(gather(r(:, k)), Row, Col).*mask;
    end
    reflImage = (reflImage-min(reflImage(:)))/(max(reflImage(:))-min(reflImage(:)));
    figure, imshow(reflImage);
    score3 = LMSE( mean(gt_refl, 3), mean(reflImage,3), mask)
    derived_shad = specImg./(reflImage+eps);
    derived_shad = (derived_shad-min(derived_shad(:)))/(max(derived_shad(:))-min(derived_shad(:)));
    score4 = LMSE( mean(gt_shad, 3), mean(derived_shad,3), mask)
    shadmat(idxI, idxJ, name) = min(score, score4);
    refmat(idxI, idxJ, name) = min(score2, score3);
    end
    idxJ = 0;
end
    imwrite(reflImage, strcat(FILENAME, '_refl', num2str(score3),'.png'));
    imwrite(derived_shad, strcat(FILENAME, '_dershad', num2str(score4),'.png'));

    
%% iteration
% % lambda_rc = 2;
% % lambda_data = 1;%0.01;
% idx = 1;
% %-------------------isolate illumination---------------
% % Bsmat = repmat(Bs(:,1)', Row, 1);
% % Bsmat = repmat(Bsmat, [1, 1, Col]);
% % Bsmat = permute(Bsmat, [1, 3, 2]);
% % gt_shad = gt_shad./Bsmat;
% % gt_shad  =mean(gt_shad, 3);
% % figure, imshow(gt_shad, []);
% % figure
% % LMSE( gt_shad, mean(shadImg,3), mask)
% while LMSE( mean(gt_shad, 3), mean(shadImg,3), mask) > 0.001
%     
%     
%     COST_FUCTION = lambda_sc*norm(Wsc*shadVec) + lambda_rc*norm(Vrc_L*shadVec)+...
%          lambda_sc*norm(Wsc_L*reflVec) + lambda_rc*norm(Vrc*reflVec)+...
%          lambda_data*norm((sparseBs*shadVec).*(sparseBr*reflVec)-reshape(specImgVec', [], 1))
% 
% %     Solve = {'solve S', 'solve R'};
% %     solveWhat = Solve(mod(idx, 2)+2)
%     
%     if mod(idx, 2)
%         % Qsc
% 
%         Q = lambda_sc*(Wsc'*Wsc);
% 
%         % Qrc
% 
%         Q = Q + lambda_rc*(Vrc_L'*Vrc_L);
%     
%     else
%         
%         % Qsc
% 
%         Q = lambda_sc*(Wsc_L'*Wsc_L);
% 
%         % Qrc
% 
%         Q = Q + lambda_rc*(Vrc'*Vrc);
% 
%     end
% 
%     % Qdata
%     C = reshape(specImgVec', [], 1);
% 
%     if mod(idx, 2)
%     % Edata-s term
%     reflVec_restore = sparseBr * reflVec; % reflVec length: N*Jr
%     Qdata = diag(sparse(reflVec_restore))*sparseBs;
% 
%     else
%     % Edata-r term
%     shadVec_restore = sparseBs * shadVec; % shadVec length: N*Js
%     Qdata = diag(sparse(shadVec_restore))*sparseBr;
% 
%     end
% 
%     Q = Q + lambda_data*(Qdata'*Qdata);
%     b = lambda_data*Qdata'*C;
%     
%     if mod(idx, 2)
%         tic,
%         TOL=1e-8;
%         [shadVec,flag,relres,iter]=pcg(Q,b,TOL,5000);
%         flag
%         relres
%         iter
%         toc
%         
%         a=reshape(shadVec,[Js,N]);
%         r=(Bs*a)';
%         
%         shadImg = zeros(Row, Col, K);
%         for k = 1:K
%             shadImg(:, :, k) = reshape(r(:, k), Row, Col).*mask;
%         end
%         imshow(shadImg);
% %         imwrite(RGB, strcat(num2str(idx),'_shading.bmp'));
%         LMSE( mean(gt_shad, 3), mean(shadImg,3), mask)
%     else
%         tic,
%         TOL=1e-8;
%         [reflVec,flag,relres,iter]=pcg(Q,b,TOL,5000);
%         flag
%         relres
%         iter
%         toc
%         
%         a=reshape(reflVec,[Jr,N]);
%         r=(Br*a)';
%         
%         reflImage = zeros(Row, Col, K);
%         for k = 1:K
%             reflImage(:, :, k) = reshape(r(:, k), Row, Col).*mask;
%         end
%     end
%     idx = idx+1;
% 
% end
%---------------------------0122
reset(gpu);
end     
save refmat.mat refmat
save shadmat.mat shadmat
