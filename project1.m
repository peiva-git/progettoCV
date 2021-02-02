imagesNumber = 20;
squareSize = 30; % 30 mm square size

%%
% load all checkerboard images

I = imread(fullfile('images', 'Image1.tif'));
[rows, cols] = size(I);
images = zeros(rows, cols, imagesNumber);

for ii=1:imagesNumber
    imageFileName = strcat('Image', string(ii), '.tif');
    images(:, :, ii) = imread(fullfile('images', imageFileName));
end

%%
% load points using code in lab1 with imageData structure array
% added board size for each image

clear imageData

for ii=1:imagesNumber
    imageFileName = fullfile('images', strcat('Image', string(ii), '.tif'));
    imageData(ii).image = imread(imageFileName);
    
    [imagePoints, boardSize] = detectCheckerboardPoints(imageData(ii).image);
    imageData(ii).XYpixels = imagePoints;
    imageData(ii).boardSize = boardSize;
end

%%
% establish correspondences, considering square size

for ii=1:imagesNumber
%    figure
%    imshow(imageData(ii).image, 'InitialMagnification', 300)
    
    clear Xmm Ymm
    for jj=1:length(imageData(ii).XYpixels)
        [row, col] = ind2sub([imageData(ii).boardSize(1) - 1, imageData(ii).boardSize(2) - 1], jj);
        Xmm = (col - 1) * squareSize;
        Ymm = (row - 1) * squareSize;
        
        imageData(ii).XYmm(jj, :) = [Xmm, Ymm];
%{        
        hndtxt = text(imageData(ii).XYpixels(jj, 1),...
            imageData(ii).XYpixels(jj, 2),...
            num2str([Xmm, Ymm]));
        set(hndtxt, 'fontsize', 14, 'color', [1 mod(row, 2) * mod(col, 2) 0]);
%}        
    end 
%    pause(1)
end

%%
% zhang method applied knowing homography
% homography estimation provided in lab1, esimate now

for ii=1:imagesNumber
    XYpixels = imageData(ii).XYpixels;
    XYmm = imageData(ii).XYmm;
    A = [];
%    b = [];
    
    for jj=1:length(XYpixels)
        
        Xpixels = XYpixels(jj, 1);
        Ypixels = XYpixels(jj, 2);
        Xmm = XYmm(jj, 1);
        Ymm = XYmm(jj, 2);
        
        m = [Xmm; Ymm; 1];
        zero = [0; 0; 0];
        A = [A; m' zero' -Xpixels*m'; zero' m' -Ypixels*m'];
%        b = [b; 0; 0];
        
    end
    
    [U, S, V] = svd(A);
    h = V(:, end);
    
    imageData(ii).H = reshape(h, [3 3])';
    
end
%%
% check if it works, superimpose rectangle
% from lab1

for ii=1:imagesNumber
    figure
    imshow(imageData(ii).image, 'InitialMagnification', 200)
    hold on
    
    width = 150;
    height = 120;
    
    topLeftCornerX = 90;
    topLeftCornerY = 60;
    
    rectLengthX = topLeftCornerX + [0 0 width width];
    rectLengthY = topLeftCornerY + [0 height height 0];
    
    homogeneous = [rectLengthX; rectLengthY; ones(1, length(rectLengthX))];
    homProjection = imageData(ii).H * homogeneous;
    
    projection = [homProjection(1, :)./homProjection(3, :);...
        homProjection(2, :)./homProjection(3, :)];
    
    projection(:, end + 1) = projection(:, 1);
    
    plot(projection(1, :), projection(2, :), 'r', 'LineWidth', 3);
    pause(1)
end
%%
% POINT 1
% Zhang method, obtain b vector (L2-p73)

V = [];

for ii=1:imagesNumber
    currentH = imageData(ii).H;
    
    V = [V; compute_v_ij(1, 2, currentH)';...
        (compute_v_ij(1, 1,  currentH) - compute_v_ij(2, 2, currentH))']; 
end

[U, D, S] = svd(V);
b = S(:, end);

% need to divide to have positive definite B? (defined up to scale factor)
b = b/b(6);

% now to build B matrix (L2-p73)

B = [b(1) b(2) b(4); b(2) b(3) b(5); b(4) b(5) b(6)];
L = chol(B, 'lower');
K = inv(L');

% set proper scale
K = K/K(3, 3);

% extrinsic parameters are computed for each image (L2-p73)
for ii=1:imagesNumber
    currentH = imageData(ii).H;
    lambda = 1/norm(K \ currentH(:, 1)); % using of inv discuraged by matlab
    
    r_1 = lambda * K \ currentH(:, 1);
    r_2 = lambda * K \ currentH(:, 2);
    R = [r_1, r_2, cross(r_1, r_2)];
    
    % find closest orthogonal matrix in Frobenius norm
    [U, S, V] = svd(R);
    R_orthogonal = U * V';
    
    imageData(ii).R = R;
    imageData(ii).R_orthogonal = R_orthogonal;
    imageData(ii).t = lambda * K \ currentH(:, 3);
    imageData(ii).K = K;
    imageData(ii).P = K * [imageData(ii).R, imageData(ii).t];
end
%%
% POINT 2
% compute and show reprojected points for chosen image
% compute total reprojection error
% get matrix P

imageIndex = 1;

% matrix P not working if using R_orthogonal
P = imageData(imageIndex).P;

figure
imshow(imageData(imageIndex).image, 'InitialMagnification', 200)
hold on

totalReprojectionError = 0;

for jj=1:length(imageData(imageIndex).XYmm)
    
    pointSpace = [imageData(imageIndex).XYmm(jj, 1);...
        imageData(imageIndex).XYmm(jj, 2); 0; 1];
    projPointX = (P(1, :) * pointSpace) / (P(3, :) * pointSpace);
    projPointY = (P(2, :) * pointSpace) / (P(3, :) * pointSpace);
    imagePointX = imageData(imageIndex).XYpixels(jj, 1);
    imagePointY = imageData(imageIndex).XYpixels(jj, 2);

    plot(imagePointX, imagePointY, 'r+')
    plot(projPointX, projPointY, 'g+')
    
    totalReprojectionError = totalReprojectionError + (projPointX - imagePointX)^2 +...
        (projPointY - imagePointY)^2;
end
%%
% POINT 3
%add radial distortion compensation 
%(Lecture 2, page 70) to the basic Zhang’s calibration procedure

%Given m and m' (correspondences) do the following:
% 1 - estimate P and get intrinsic parameters from P
% 2 - estimate k1 and k2
% 3 - compensate for radial distortion and get new m'(i) 
% 4 - go to the step 1 until convergence of P and k1 and k2

% We have already P = K [R | t]
% get intrinsic parameters and rd

iterationsCounter = 1;
maxIterations = 10;
totalErrors = zeros(maxIterations, 1);
k_vectors = zeros(imagesNumber * maxIterations, 2);

% first build linear system to estimate k using all correspondences in
% the image, for each image (need all images to estimate P again)

while iterationsCounter < maxIterations + 1
    
% first estimate P

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % copied code
    
    % estimate homography
    
    for ii=1:imagesNumber
        XYpixels = imageData(ii).XYpixels;
        XYmm = imageData(ii).XYmm;
        
        A = zeros(2 * length(XYpixels), 9);
        %    b = [];
        
        for jj=1:length(XYpixels)
            
            Xpixels = XYpixels(jj, 1);
            Ypixels = XYpixels(jj, 2);
            Xmm = XYmm(jj, 1);
            Ymm = XYmm(jj, 2);
            
            m = [Xmm; Ymm; 1];
            zero = [0; 0; 0];
            A(2 * jj - 1, :) = [m' zero' -Xpixels*m']; % odd rows
            A(2 * jj, :) = [zero' m' -Ypixels*m']; % even rows
            %        b = [b; 0; 0];
            
        end
        
        [~, ~, V] = svd(A);
        h = V(:, end);
        
        imageData(ii).H = reshape(h, [3 3])';       
    end
    
    % now find K, R and t (intrinsic and extrinsic parameters)
    
    V = zeros(2 * imagesNumber, 6);
    
    for ii=1:imagesNumber
        currentH = imageData(ii).H;
        
        V(2 * ii - 1, :) = compute_v_ij(1, 2, currentH)'; % odd rows
        V(2 * ii, :) = (compute_v_ij(1, 1,  currentH) - compute_v_ij(2, 2, currentH))'; % even rows
    end
    
    [~, ~, S] = svd(V);
    b = S(:, end);
    
    % need to divide to have positive definite B? (defined up to scale factor)
    b = b/b(6);
    
    % now to build B matrix (L2-p73)
    
    B = [b(1) b(2) b(4); b(2) b(3) b(5); b(4) b(5) b(6)];
    L = chol(B, 'lower');
    K = inv(L');
    
    % set proper scale
    K = K/K(3, 3);
    
    % extrinsic parameters are computed for each image (L2-p73)
    for ii=1:imagesNumber
        currentH = imageData(ii).H;
        lambda = 1/norm(K \ currentH(:, 1)); % using of inv discuraged by matlab
        
        r_1 = lambda * K \ currentH(:, 1);
        r_2 = lambda * K \ currentH(:, 2);
        R = [r_1, r_2, cross(r_1, r_2)];
        
        % find closest orthogonal matrix in Frobenius norm
%        [U, ~, V] = svd(R);
%        R_orthogonal = U * V';
        
        imageData(ii).R = R;
%        imageData(ii).R_orthogonal = R_orthogonal;
        imageData(ii).t = lambda * K \ currentH(:, 3);
        imageData(ii).K = K; % same for all images
        
        % finally compute matrix P
        imageData(ii).P = K * [imageData(ii).R, imageData(ii).t];
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    for ii=1:imagesNumber
        
        u_0 = imageData(ii).K(1,3);
        v_0 = imageData(ii).K(2,3);
        alpha_u = imageData(ii).K(1,1);
        skew_angle = acot(imageData(ii).K(1,2)/alpha_u); % cotan = 1/tan, inverse is acotan
        alpha_v = imageData(ii).K(2,2) * sin(skew_angle);
        
        P = imageData(ii).P;
        
        A = zeros(2 * length(imageData(ii).XYmm), 2);
        b = zeros(2 * length(imageData(ii).XYmm), 1);
        
        for jj=1:length(imageData(ii).XYmm)
            
            pointSpace = [imageData(ii).XYmm(jj, 1);...
                imageData(ii).XYmm(jj, 2); 0; 1];
            
            projPointX = (P(1, :) * pointSpace) / (P(3, :) * pointSpace); %u actual projections
            projPointY = (P(2, :) * pointSpace) / (P(3, :) * pointSpace); %v actual projections
            
            imagePointX = imageData(ii).XYpixels(jj, 1); %u^ distorted projections
            imagePointY = imageData(ii).XYpixels(jj, 2); %v^ distorted projections
            
            rd_2 = ((projPointX - u_0)/alpha_u)^2 + ((projPointY - v_0)/alpha_v)^2;
            
            A(2 * jj - 1, 1) = (projPointX - u_0) * rd_2; % odd rows
            A(2 * jj - 1, 2) = (projPointX - u_0) * rd_2 * rd_2; % odd rows
            A(2 * jj, 1) = (projPointY - v_0) * rd_2; % even rows
            A(2 * jj, 2) = (projPointY - v_0) * rd_2 * rd_2; % even rows
            
            b(2 * jj - 1, 1) = imagePointX - projPointX; % odd rows
            b(2 * jj, 1) = imagePointY - projPointY; % even rows
        end
        
        % now estimate k using least squares
        k = (A'*A)\A' * b; % k is 2x1 vector
        k_1 = k(1, 1);
        k_2 = k(2, 1);
        
        k_vectors(ii + (iterationsCounter - 1) * imagesNumber, :) = k;
        
        % now build nonlinear system to compensate distortion
        
        for jj=1:length(imageData(ii).XYmm)
            
            clear nonLinearCompensation coord x0 % clear from previous step
            nonlinearCompensation = eqnproblem; % optimization toolbox
            coord = optimvar('coord', 2);
            
            pointSpace = [imageData(ii).XYmm(jj, 1);...
                imageData(ii).XYmm(jj, 2); 0; 1];
            
            projPointX = (P(1, :) * pointSpace) / (P(3, :) * pointSpace); %u actual projections
            projPointY = (P(2, :) * pointSpace) / (P(3, :) * pointSpace); %v actual projections
            
            pointProjX = (projPointX - u_0) / alpha_u; % x actual coordinates
            pointProjY = (projPointY - v_0) / alpha_v; % y actual coordinates
            
            imagePointX = imageData(ii).XYpixels(jj, 1); % u^
            imagePointY = imageData(ii).XYpixels(jj, 2); % v^
            
            pointDistortedX = (imagePointX - u_0) / alpha_u; % x^ distorted coordinates
            pointDistortedY = (imagePointY - v_0) / alpha_v; % y^ distorted coordinates
            
            equation_1 = coord(1) * (1 + k_1 * (coord(1)^2 + coord(2)^2) + k_2 * (coord(1)^4 + 2 * (coord(1)^2) * (coord(2)^2) + coord(2)^4)) - pointDistortedX == 0;
            equation_2 = coord(2) * (1 + k_1 * (coord(1)^2 + coord(2)^2) + k_2 * (coord(1)^4 + 2 * (coord(1)^2) * (coord(2)^2) + coord(2)^4)) - pointDistortedY == 0;
            
            nonlinearCompensation.Equations.equation_1 = equation_1;
            nonlinearCompensation.Equations.equation_2 = equation_2;
            
            % solve for each pair of coordinates
            
            x0.coord = [pointProjX pointProjY]; % search close to actual values
            
            [sol, ~, ~] = solve(nonlinearCompensation , x0);
            
            % store new compensated coordinates
            % use same variable, values will be now reused to estimate P again
            imageData(ii).XYpixels(jj, 1) = alpha_u * sol.coord(1) + u_0;
            imageData(ii).XYpixels(jj, 2) = alpha_v * sol.coord(2) + v_0;
            
            % TODO compute new P with compensated coordinates
            % iterate, using new coordinates with matrix P
        end
    end
    
    % check error only on one image
    
    totalReprojectionError = 0;
    
    figure
    imshow(imageData(imageIndex).image, 'InitialMagnification', 200)
    hold on
    
    % get P matrix of chosen image
    P_plot = imageData(imageIndex).P;
    
    for jj=1:length(imageData(imageIndex).XYmm)
    
        pointSpace = [imageData(imageIndex).XYmm(jj, 1);...
            imageData(imageIndex).XYmm(jj, 2); 0; 1];
        projPointX = (P_plot(1, :) * pointSpace) / (P_plot(3, :) * pointSpace);
        projPointY = (P_plot(2, :) * pointSpace) / (P_plot(3, :) * pointSpace);
        imagePointX = imageData(imageIndex).XYpixels(jj, 1);
        imagePointY = imageData(imageIndex).XYpixels(jj, 2);
        
        plot(imagePointX, imagePointY, 'r+')
        plot(projPointX, projPointY, 'g+')

        totalReprojectionError = totalReprojectionError +...
            (projPointX - imagePointX)^2 +...
            (projPointY - imagePointY)^2;
    end
    
    totalErrors(iterationsCounter, 1) = totalReprojectionError;
    
    iterationsCounter = iterationsCounter + 1;
end
%%
% trying problem - based approach to solve nonlinear system of equations
% using optimization toolbox

x = optimvar('x', 2);
first_eq = x(1)^2 + x(2)^2 == 1;
second_eq = x(1) + x(2) == 0;
x0.x = [1 -1]; % must be sufficiently close to find solution
problem = eqnproblem;

problem.Equations.first_eq = first_eq;
problem.Equations.second_eq = second_eq;
show(problem);

[sol, fvak, exitflag] = solve(problem , x0); % correct solution found


 



