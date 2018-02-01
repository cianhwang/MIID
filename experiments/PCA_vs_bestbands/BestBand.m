clc;clear;close all

addpath ('../toolkit', '../toolkit/spec2rgb', '../toolkit/load_data')
nameList = { 'box', 'car', 'cup', 'train', 'vase', 'dinosaur', 'minion', 'plane' ,'bus','car2','vase2', 'vase3'};

for nameIdx = 1:length(namelist)
    FILENAME = char(nameList(nameIdx));
    addpath(strcat('../GroundTruth/450-700nm_118通道红头灯/',FILENAME));

%% read spectral data
load('diffusewithill');
load('reflectance');
load('shadingwithill');
gt_shad = shadingwithill(:, :, 1:4:end); 
specImg = diffusewithill(:, :, 1:4:end); 
gt_refl =  specImg./(gt_shad+eps);

mask = imread('mask.bmp');
mask = mean(mask, 3);
mask = logical(mask);
% groundtruth;

specImgRGB = im2double(imread('diffuse.bmp'));

%---------------block--------------
[mask, specImg, specImgRGB, gt_shad, gt_refl] = RESIZE(0.75, mask, specImg, specImgRGB, gt_shad, gt_refl);

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

RGB = convRGB(specImg, 450, 700, (700-450)/(K-1));
figure, imshow(RGB);    


%% Bs, Br
%-----------Br---------------
Jr = 8;
load('450to700_118chan_basis.mat');
coeff = pca(SPECBASIS(:, 1:4:end));
% coeff = pca(specImgVec);
Br = coeff(:, 1:Jr);
%-------------20180121----------------
Br = BestBandMap(specImg, Jr);

% Jr = 3; Br = eye(3);

%-----------Bs---------------
Js = 1;
load('illumination');
Bs = illuminationREDHEAD(1:4:end)';
Bs = Bs/norm(Bs);

% shadVec = reshape(shading, [], 59);
% coeff = pca(shadVec);
% Js = 2; Bs = coeff(1:2:59, 1:Js);

% Js = 1; Bs = ones(K, 1)/(K^0.5);

%% Weightmap
H = Normalized(specImg);
beta  = 1e-3;%2.636650898730358e-04;
alpha = 5000;
[Wr, Wd] = weightMap(specImg, alpha, beta, 0); %xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
    for i = 1:1:N % 行
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
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
lambda_sc = 1;%00000;
lambda_rc = 2;%00000;%2.559547922699536e+02;
lambda_gc = 0;%.1;
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    %% S estimate

    lambda_esti_s = 0.005;%0.01;%0.0005;%0.00138949549437300;
%     Wsc = sparse(rowIndex, colIndex, WscVal, N*NeighSize*K, N*K)*sparseBs;
%     Vrc_L = sparse(rowIndex, colIndex, Vrc_LVal, N*NeighSize*K, N*K)*sparseBs;
%     Gcs = Gc*sparseBs;

    % Generic_s
    %---------------------------
    L = speye(N*K);     L = L*sparseBs;
    C = reshape(specImgVec', [], 1);
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
        
    b = lambda_esti_s*L'*C;
    Q = lambda_sc*(Wsc'*Wsc) + lambda_rc*(Vrc_L'*Vrc_L) +lambda_esti_s*(L'*L);%+ lambda_gc*(Gc'*Gc);
    disp('Q s-estimate complete!')
    
    tic,
    TOL=1e-8;
    [shadVec,flag,relres,iter]=pcg(Q,b,TOL,2000);
    flag
    relres
    iter
    toc
%     specImgLongVector =specImgVec'; specImgLongVector = sparseBs'*specImgLongVector(:);
% %     lambda_max = norm(cat(1,lambda_sc*Wsc,lambda_rc*Vrc_L)'*cat(1,-lambda_sc*Wsc*specImgLongVector,-lambda_rc*Vrc_L*specImgLongVector), 'inf');
% %     lambda = 0.1*lambda_max;
% %     [shadVec history] = lasso(Wsc+Vrc_L, -Wsc*specImgLongVector-Vrc_L*specImgLongVector, lambda, 1.0, 1.0);
% 	[shadVec history] = lasso(cat(1,lambda_sc*Wsc,lambda_rc*Vrc_L), cat(1,-lambda_sc*Wsc*specImgLongVector,-lambda_rc*Vrc_L*specImgLongVector), lambda_esti_s, 1.0, 1.0);
%     shadVec = specImgLongVector+shadVec;
%     [shadVec history] = lad(cat(1, lambda_sc*Wsc, lambda_rc*Vrc_L, lambda_esti_s*L ),...
%         cat(1, zeros(N*NeighSize*K, 1), zeros(N*NeighSize*K, 1), lambda_esti_s*C), 1, 1.0);

    
    a=reshape(shadVec,[Js,N]);
    r=(Bs*a)';
    
    shadImg = zeros(Row, Col, K);
    for k = 1:K
        shadImg(:, :, k) = reshape(r(:, k), Row, Col).*mask;
    end

    score = LMSE(gt_shad,shadImg, mask)
    
    derived_reflImg = specImg./(shadImg+eps);
    score = LMSE( gt_refl,derived_reflImg, mask);

%% E r-term

    %---------------------sparseBs-----------------------


    Br_temp = sparse(Br);
    Br_temp = repmat({Br_temp}, N, 1);
    sparseBr = blkdiag(Br_temp{:});
  


    % R estimate
    Wsc_L = sparse(rowIndex, colIndex, Wsc_LVal, N*NeighSize*K, N*K)*sparseBr;
    Vrc = sparse(rowIndex, colIndex, VrcVal, N*NeighSize*K, N*K)*sparseBr;
    
    lambda_esti_r = 0.005;%0.005;
    lambda_data = 1;
	% Generic_r
    %---------------------------
    L = speye(N*K); L = L*sparseBr;
    C = reshape(specImgVec', [], 1);
    %----------------------------------
%     L=zeros(1,N*K);
%     C=zeros(1,1);
%     for j = 1:2:N
%         for k = 1
%             L(k,(j-1)*K+k)=mask(j);
%             C(k) =  C(k) + specImgVec(j, k);
%         end
%     end
%     L = sparse(L)*sparseBr;

    b = lambda_esti_r*L'*C;
    Q = lambda_sc*(Wsc_L'*Wsc_L) + lambda_rc*(Vrc'*Vrc) +lambda_esti_r*(L'*L);
    
    shadVec_restore = sparseBs*shadVec;
    Qdata = diag(sparse(shadVec_restore)) * sparseBr;
    C = reshape(specImgVec', [], 1);
    
    Q = Q + lambda_data*(Qdata'*Qdata);
    b = b + lambda_data*Qdata'*C;
    disp('Q r-estimate complete!')

    tic,
    TOL=1e-8;
    [reflVec,flag,relres,iter]=pcg(Q,b,TOL,2000);
    flag
    relres
    iter
	toc
    
    a=reshape(reflVec,[Jr,N]);
    r=(Br*a)';
    
    reflImage = zeros(Row, Col, K);
    for k = 1:K
        reflImage(:, :, k) = reshape(r(:, k), Row, Col).*mask;
    end

    score2= LMSE( gt_refl,reflImage, mask)

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
% figure
% while idx < 1000
%     if mod(idx, 2)
%         lambda_data = 0.01;
%     else
%         lambda_data = 1;
%     end
% 
%     
%     
%    E_SUM(idx)=lambda_sc*norm(Wsc*shadVec) + lambda_rc*norm(Vrc_L*shadVec)+...
%          lambda_sc*norm(Wsc_L*reflVec) + lambda_rc*norm(Vrc*reflVec)+...
%          lambda_data*norm((sparseBs*shadVec).*(sparseBr*reflVec)-reshape(specImgVec', [], 1));
%      if idx > 3
%          delta = -E_SUM(idx) +E_SUM(idx-2);
%          if abs(delta) < 0.01
%              break;
%          end
%      end
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
%         TOL=1e-6;
%         [shadVec,flag,relres,iter]=pcg(Q,b,TOL,1000);
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
%         RGB = convRGB(shadImg, 400, 700, (700-400)/(K-1));
%         imshow(RGB);
%         imwrite(RGB, strcat(num2str(idx),'_shading.bmp'));
%         score(idx) = LMSE(gt_shad,shadImg, mask)
%         imwrite(RGB, strcat(FILENAME, '_shading', num2str(score),'.png'));
% 
%     else
%         tic,
%         TOL=1e-6;
%         [reflVec,flag,relres,iter]=pcg(Q,b,TOL,1000);
%         flag
%         relres
%         iter
%         toc
%         
% %         a=reshape(reflVec,[Jr,N]);
% %         r=(Br*a)';
% %         
% %         reflImage = zeros(Row, Col, K);
% %         for k = 1:K
% %             reflImage(:, :, k) = reshape(r(:, k), Row, Col).*mask;
% %         end
%     end
%     idx = idx+1;
% 
% end
end