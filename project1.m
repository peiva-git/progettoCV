imagesNumber = 20; % set to 10 if using author's images
squareSize = 30; % 30 mm square size, 22 if using author's images

global checkNormalizedX; % used inside function
global checkNormalizedY;
global k_1; % used inside function
global k_2;

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

clear imageData

for ii=1:imagesNumber
    imageFileName = fullfile('images', strcat('Image', string(ii), '.tif'));
    imageData(ii).image = imread(imageFileName);
    
    [imagePoints, boardSize] = detectCheckerboardPoints(imageData(ii).image);
    imageData(ii).XYpixels = imagePoints;
    imageData(ii).checkerboardPixels = imagePoints;
    imageData(ii).boardSize = boardSize;
end
%%
% establish correspondences, considering square size

for ii=1:imagesNumber
    
    clear Xmm Ymm
    for jj=1:length(imageData(ii).XYpixels)
        [row, col] = ind2sub([imageData(ii).boardSize(1) - 1, imageData(ii).boardSize(2) - 1], jj);
        Xmm = (col - 1) * squareSize;
        Ymm = (row - 1) * squareSize;
        
        imageData(ii).XYmm(jj, :) = [Xmm, Ymm];       
    end 
end
%%
% zhang method applied knowing homography
% homography estimation provided in lab1, esimate now

for ii=1:imagesNumber
    XYpixels = imageData(ii).XYpixels;
    XYmm = imageData(ii).XYmm;
    A = zeros(2 * length(XYpixels), 9);
    
    for jj=1:length(XYpixels)
        
        Xpixels = XYpixels(jj, 1);
        Ypixels = XYpixels(jj, 2);
        Xmm = XYmm(jj, 1);
        Ymm = XYmm(jj, 2);
        
        m = [Xmm; Ymm; 1];
        zero = [0; 0; 0];
        A(2 * jj - 1, :) = [m' zero' -Xpixels*m']; % odd rows
        A(2 * jj, :) = [zero' m' -Ypixels*m']; % even rows       
    end
    
    [~, ~, V] = svd(A);
    h = V(:, end);
    
    imageData(ii).H = reshape(h, [3 3])';
    
    % change det to prevent flipped z axis
    imageData(ii).H = nthroot(- 1/det(imageData(ii).H), 3) * imageData(ii).H;
    
end
%%
% POINT 1
% Zhang's method, obtain b vector (L2-p73)

V = zeros(2 * imagesNumber, 6);

for ii=1:imagesNumber
    currentH = imageData(ii).H;
    
    V(2 * ii - 1, :) = compute_v_ij(1, 2, currentH)'; % odd rows
    V(2 * ii, :) = (compute_v_ij(1, 1,  currentH) - compute_v_ij(2, 2, currentH))'; % even rows 
end

[~, ~, S] = svd(V);
b = S(:, end);

% divide to have positive definite B (defined up to scale factor)
b = b/b(6);

% now build B matrix (L2-p73)

B = [b(1) b(2) b(4); b(2) b(3) b(5); b(4) b(5) b(6)];
L = chol(B, 'lower');
K = inv(L');

% set proper scale
K = K/K(3, 3);

% extrinsic parameters are computed for each image (L2-p73)
for ii=1:imagesNumber
    currentH = imageData(ii).H;
    lambda = 1/norm(K \ currentH(:, 1)); % using of inv discuraged by matlab
    
    r_1 = lambda * (K \ currentH(:, 1));
    r_2 = lambda * (K \ currentH(:, 2));
    R = [r_1, r_2, cross(r_1, r_2)];
    
    % find closest orthogonal matrix in Frobenius norm
    [U, ~, V] = svd(R);
    R_orthogonal = U * V';
    
    imageData(ii).R = R;
    imageData(ii).R_orthogonal = R_orthogonal;
    imageData(ii).t = lambda * (K \ currentH(:, 3));
    imageData(ii).K = K;
    imageData(ii).P = K * [imageData(ii).R, imageData(ii).t];
end
%%
% POINT 2
% obtain projection matrix P
% compute and show reprojected points for chosen image
% compute total reprojection error for chosen image

imageIndex = 1;

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

    plot(imagePointX, imagePointY, 'r+', 'MarkerSize', 8)
    plot(projPointX, projPointY, 'g+', 'MarkerSize', 8)
    
    totalReprojectionError = totalReprojectionError + (projPointX - imagePointX)^2 +...
        (projPointY - imagePointY)^2;
end
%%
% POINT 3
% add radial distortion compensation 
% (L2-p70) to the basic Zhang’s calibration procedure

% given m and m' (correspondences) do the following:
% 1 - estimate P and get intrinsic parameters from P
% 2 - estimate k1 and k2
% 3 - compensate for radial distortion and get new m'(i) 
% 4 - go to the step 1 until convergence of P and k1 and k2

% specify number of iterations
maxIterations = 50;

iterationsCounter = 1;

totalErrors_reprojection = zeros(maxIterations, 1);
totalErrors_k = zeros(maxIterations - 1, 1);
k_vectors = zeros(imagesNumber * maxIterations, 2);

exitflags = zeros(imagesNumber * maxIterations * length(imageData(imageIndex).XYmm), 1);
outputs = cell(imagesNumber * maxIterations * length(imageData(imageIndex).XYmm), 1);
solutions = cell(imagesNumber * maxIterations * length(imageData(imageIndex).XYmm), 1);

while iterationsCounter < maxIterations + 1
    
% first estimate P

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % copied code
    
    % estimate homography
    
    for ii=1:imagesNumber
        XYpixels = imageData(ii).XYpixels;
        XYmm = imageData(ii).XYmm;
        
        A = zeros(2 * length(XYpixels), 9);
        
        for jj=1:length(XYpixels)
            
            Xpixels = XYpixels(jj, 1);
            Ypixels = XYpixels(jj, 2);
            Xmm = XYmm(jj, 1);
            Ymm = XYmm(jj, 2);
            
            m = [Xmm; Ymm; 1];
            zero = [0; 0; 0];
            A(2 * jj - 1, :) = [m' zero' -Xpixels*m']; % odd rows
            A(2 * jj, :) = [zero' m' -Ypixels*m']; % even rows
            
        end
        
        [~, ~, V] = svd(A);
        h = V(:, end);
        
        imageData(ii).H = reshape(h, [3 3])';  
        imageData(ii).H = nthroot(- 1/det(imageData(ii).H), 3) * imageData(ii).H;
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
    
    % need to divide to have positive definite B (defined up to scale factor)
    b = b/b(6);
    
    % now build B matrix (L2-p73)
    
    B = [b(1) b(2) b(4); b(2) b(3) b(5); b(4) b(5) b(6)];
    L = chol(B, 'lower');
    K = inv(L');
    
    % set proper scale
    K = K/K(3, 3);
    
    % extrinsic parameters are computed for each image (L2-p73)
    for ii=1:imagesNumber
        currentH = imageData(ii).H;
        lambda = 1/norm(K \ currentH(:, 1)); % using of inv discuraged by matlab
        
        r_1 = lambda * (K \ currentH(:, 1));
        r_2 = lambda * (K \ currentH(:, 2));
        R = [r_1, r_2, cross(r_1, r_2)];
        
        % find closest orthogonal matrix in Frobenius norm
        [U, ~, V] = svd(R);
        R_orthogonal = U * V';
        
        imageData(ii).R = R;
        imageData(ii).R_orthogonal = R_orthogonal;
        imageData(ii).t = lambda * (K \ currentH(:, 3));
        imageData(ii).K = K; % same for all images
        
        % finally compute matrix P
        imageData(ii).P = K * [imageData(ii).R, imageData(ii).t];
    end
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% for each image, estimate k and then compensate for radial distortion

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
            
            pointCheckerboardX = imageData(ii).checkerboardPixels(jj, 1); % u^ effective distorted projections
            pointCheckerboardY = imageData(ii).checkerboardPixels(jj, 2); % v^ effective distorted projections
            
            rd_2 = ((projPointX - u_0)/alpha_u)^2 + ((projPointY - v_0)/alpha_v)^2;
            
            % build linear system using all correspondences
            
            A(2 * jj - 1, 1) = (projPointX - u_0) * rd_2; % odd rows
            A(2 * jj - 1, 2) = (projPointX - u_0) * rd_2 * rd_2; % odd rows
            A(2 * jj, 1) = (projPointY - v_0) * rd_2; % even rows
            A(2 * jj, 2) = (projPointY - v_0) * rd_2 * rd_2; % even rows
            
            b(2 * jj - 1, 1) = pointCheckerboardX - projPointX; % odd rows
            b(2 * jj, 1) = pointCheckerboardY - projPointY; % even rows
        end
        
        % now estimate k using least squares
        k = (A' * A)\A' * b; % k is 2x1 vector
        k_1 = k(1, 1);
        k_2 = k(2, 1);
        imageData(ii).k_1 = k_1;
        imageData(ii).k_2 = k_2;
        
        % save all values of k
        k_vectors(ii + (iterationsCounter - 1) * imagesNumber, 1) = k_1;
        k_vectors(ii + (iterationsCounter - 1) * imagesNumber, 2) = k_2;
        
        % now build nonlinear system to compensate distortion
        % for each correspondence
        
        for jj=1:length(imageData(ii).XYmm)
                
            pointCheckerboardX = imageData(ii).checkerboardPixels(jj, 1); % u^ effective distorted coordinates
            pointCheckerboardY = imageData(ii).checkerboardPixels(jj, 2); % v^ effective distorted coordinates
            
            pointSpace = [imageData(ii).XYmm(jj, 1);...
                imageData(ii).XYmm(jj, 2); 0; 1];
            
            projPointX = (P(1, :) * pointSpace) / (P(3, :) * pointSpace); %u actual projections
            projPointY = (P(2, :) * pointSpace) / (P(3, :) * pointSpace); %v actual projections
            
            pointProjX = (projPointX - u_0) / alpha_u; % x actual coordinates
            pointProjY = (projPointY - v_0) / alpha_v; % y actual coordinates
            
            checkNormalizedX = (pointCheckerboardX - u_0) / alpha_u; % normalized, used inside function
            checkNormalizedY = (pointCheckerboardY - v_0) / alpha_v; % normalized
            
            % solve for each pair of coordinates
            
            x0 = [pointProjX; pointProjY]; % search close to actual values
            
            % solution with fsolve
            % use Parallel Computing Toolbox and Jacobian specified in
            % @distortionCompensation function
            opts = optimoptions('fsolve', 'UseParallel', true, 'SpecifyObjectiveGradient', true, 'Display', 'off');
            [sol, ~, exitflag, output, ~] = fsolve(@distortionCompensation, x0, opts);
            
            exitflags(jj + length(imageData(ii).XYmm) * ii * (iterationsCounter - 1), 1) = exitflag;
            outputs{jj + length(imageData(ii).XYmm) * ii * (iterationsCounter - 1), 1} = output;
            solutions{jj + length(imageData(ii).XYmm) * ii * (iterationsCounter - 1), 1} = sol;
            
            % solution found, assign new value
            if (exitflag > 0)
            
                u_sol = alpha_u * sol(1) + u_0;
                v_sol = alpha_v * sol(2) + v_0;
            
                % store new theoretical projections
                % use same variable, values will be reused to estimate P again
                imageData(ii).XYpixels(jj, 1) = u_sol;
                imageData(ii).XYpixels(jj, 2) = v_sol;
                
            % solution not found, keep previous value, print message
            elseif (exitflag == 0)
                    fprintf('Exceeded max evaluations, func evaluations: %u, iterations: %u\nError message: %s\n', output.funcCount, output.iterations, output.message)
            else
                fprintf('%s\n', output.message)
            end
            
            % iterate, estimate new P
        end
    end
    
    % check error and convergence only on one image
    
    totalReprojectionError = 0;
    
    u_0 = imageData(imageIndex).K(1,3);
    v_0 = imageData(imageIndex).K(2,3);
    alpha_u = imageData(imageIndex).K(1,1);
    skew_angle = acot(imageData(imageIndex).K(1,2)/alpha_u); % cotan = 1/tan, inverse is acotan
    alpha_v = imageData(imageIndex).K(2,2) * sin(skew_angle);
    k_1 =  imageData(imageIndex).k_1;
    k_2 = imageData(imageIndex).k_2;
    P_plot = imageData(imageIndex).P;
    
    if (maxIterations > 10)
        if (mod(iterationsCounter, 10) == 0) 
            figure
            imshow(imageData(imageIndex).image, 'InitialMagnification', 200)
            hold on
        end
    else
        figure
        imshow(imageData(imageIndex).image, 'InitialMagnification', 200)
        hold on
    end
    
    % plot reprojected point for chosen image and compute error
    for jj=1:length(imageData(imageIndex).XYmm)
    
        pointSpace = [imageData(imageIndex).XYmm(jj, 1);...
            imageData(imageIndex).XYmm(jj, 2); 0; 1];
        projPointX = (P_plot(1, :) * pointSpace) / (P_plot(3, :) * pointSpace);
        projPointY = (P_plot(2, :) * pointSpace) / (P_plot(3, :) * pointSpace);
        
        % add radial distortion compensation
        rd_2 = ((projPointX - u_0)/alpha_u)^2 + ((projPointY - v_0)/alpha_v)^2;
        compensatedX = (projPointX - u_0) * (1 + k_1 * rd_2 + k_2 * rd_2 * rd_2) + u_0;
        compensatedY = (projPointY - v_0) * (1 + k_1 * rd_2 + k_2 * rd_2 * rd_2) + v_0;
        
        imagePointX = imageData(imageIndex).checkerboardPixels(jj, 1);
        imagePointY = imageData(imageIndex).checkerboardPixels(jj, 2);
        
        if (maxIterations > 10)
            if (mod(iterationsCounter, 10) == 0)
                plot(imagePointX, imagePointY, 'r+', 'MarkerSize', 8)
                plot(compensatedX, compensatedY, 'g+', 'MarkerSize', 8)
            end
        else
            plot(imagePointX, imagePointY, 'r+', 'MarkerSize', 8)
            plot(compensatedX, compensatedY, 'g+', 'MarkerSize', 8)
        end

        totalReprojectionError = totalReprojectionError +...
            (compensatedX - imagePointX)^2 +...
            (compensatedY - imagePointY)^2;
    end
    
    totalErrors_reprojection(iterationsCounter, 1) = totalReprojectionError;
    
    % consider difference in k_1 and k_2 between current and previous
    % iteration for chosen image
    if (iterationsCounter > 1)
        
        totalErrors_k(iterationsCounter - 1, 1) = (k_vectors(imageIndex + imagesNumber * (iterationsCounter - 1), 1) - k_vectors(imageIndex + imagesNumber * (iterationsCounter - 2), 1))^2 +...
            (k_vectors(imageIndex + imagesNumber * (iterationsCounter - 1), 2) - k_vectors(imageIndex + imagesNumber * (iterationsCounter - 2), 2))^2;
    end
    
    iterationsCounter = iterationsCounter + 1;
    
    pause(1)
end

figure
plot(totalErrors_reprojection)

figure
plot(totalErrors_k)
%%
% superimpose cylinder

r = 120; % in mm, 30 if using author's images
h = 60; % in mm, 30 if using author's images 
x = 150; % in mm, 60 if using author's images
y = 150; % in mm, 60 if using author's images

[X, Y, Z] = cylinder(r);

X = X + x;
Y = Y + y;
Z = Z * h;

for ii=1:imagesNumber
    
    figure
    imshow(imageData(ii).image, 'InitialMagnification', 200)
    hold on
    
    P_plot = imageData(ii).P;
    u_0 = imageData(ii).K(1,3);
    v_0 = imageData(ii).K(2,3);
    alpha_u = imageData(ii).K(1,1);
    skew_angle = acot(imageData(ii).K(1,2)/alpha_u); % cotan = 1/tan, inverse is acotan
    alpha_v = imageData(ii).K(2,2) * sin(skew_angle);
    k_1 =  imageData(ii).k_1;
    k_2 = imageData(ii).k_2;
    
    % superimpose two sets of points on image
    % first project cylinder points
    spacePoints_1 = [X(1, :); Y(1, :); Z(1, :); ones(1, length(X(1, :)))]; % 4x21 matrix
    spacePoints_2 = [X(2, :); Y(2, :); Z(2, :); ones(1, length(X(2, :)))];
    
    homProjection_1 = P_plot * spacePoints_1;
    homProjection_2 = P_plot * spacePoints_2;
    projection_1 = [homProjection_1(1, :)./homProjection_1(3, :);...
        homProjection_1(2, :)./homProjection_1(3, :)];
    projection_2 = [homProjection_2(1, :) ./ homProjection_2(3, :);...
        homProjection_2(2, :) ./ homProjection_2(3, :)];
    
    compensatedPoints_1 = zeros(2, length(projection_1));
    compensatedPoints_2 = zeros(2, length(projection_2));
    
    % apply radial distortion compensation
    for kk=1:length(compensatedPoints_1)
        rd_2 = ((projection_1(1, kk) - u_0)/alpha_u)^2 + ((projection_1(2, kk) - v_0)/alpha_v)^2;
        compensatedPoints_1(1, kk) = (projection_1(1, kk) - u_0) * (1 + k_1 * rd_2 + k_2 * rd_2 * rd_2) + u_0;
        compensatedPoints_1(2, kk) = (projection_1(2, kk) - v_0) * (1 + k_1 * rd_2 + k_2 * rd_2 * rd_2) + v_0;
        compensatedPoints_2(1, kk) = (projection_2(1, kk) - u_0) * (1 + k_1 * rd_2 + k_2 * rd_2 * rd_2) + u_0;
        compensatedPoints_2(2, kk) = (projection_2(2, kk) - v_0) * (1 + k_1 * rd_2 + k_2 * rd_2 * rd_2) + v_0;
    end
    
    % prepare shapes to plot according to
    % https://it.mathworks.com/help/vision/ref/showshape.html
    shapes = zeros(2, 2 * length(X(1, :)));
    
    for kk=1:length(compensatedPoints_1)
        shapes(1, 2 * kk - 1) = compensatedPoints_1(1, kk); % odd columns
        shapes(1, 2 * kk) = compensatedPoints_1(2, kk); % even columns
    end
    
    for kk=1:length(compensatedPoints_2)
        shapes(2, 2 * kk - 1) = compensatedPoints_2(1, kk);
        shapes(2, 2 * kk) = compensatedPoints_2(2, kk);
    end
    
    positions = cell(2, 1);
    positions{1, 1} = shapes(1, :);
    positions{2, 1} = shapes(2, :);
    
    % plot shapes on image
    showShape('polygon', positions, 'Color', {'red', 'green'}, 'Opacity', 0.7)
    pause(1)

end
%%
% solve using function and Jacobian
function [F, J] = distortionCompensation(x)
    global k_1;
    global k_2;
    global checkNormalizedX;
    global checkNormalizedY;
    
    F = zeros(2, 1);
    F(1) = x(1) * (1 + k_1 * (x(1)^2 + x(2)^2) + k_2 * (x(1)^4 + 2 * (x(1)^2) * (x(2)^2) + x(2)^4)) - checkNormalizedX;
    F(2) = x(2) * (1 + k_1 * (x(1)^2 + x(2)^2) + k_2 * (x(1)^4 + 2 * (x(1)^2) * (x(2)^2) + x(2)^4)) - checkNormalizedY;

    J = zeros(2, 2);
    J(1, 1) = 1 + 3 * k_1 * (x(1)^2) + k_1 * (x(2)^2) + 5 * k_2 * (x(1)^4) + 6 * k_2 * (x(1)^2) * (x(2)^2) + k_2 * (x(2)^4);
    J(1, 2) = 2 * k_1 * x(1) * x(2) + 4 * k_2 * (x(1)^3) * x(2) + 4 * k_2 * x(1) * (x(2)^3);
    J(2, 1) = 2 * k_1 * x(1) * x(2) + 4 * k_2 * (x(1)^3) * x(2) + 4 * k_2 * x(1) * (x(2)^3);
    J(2, 2) = 1 + 3 * k_1 * (x(2)^2) + k_1 * (x(1)^2) + 5 * k_2 * (x(2)^4) + 6 * k_2 * (x(1)^2) * (x(2)^2) + k_2 * (x(1)^4);
end