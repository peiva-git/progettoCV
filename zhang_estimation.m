function new_imageData = zhang_estimation(imageData, imagesNumber)
    
    % code copied from project1
    % XYmm and XYpixels must be previously evaluated
    
    % preallocate new struct
    
    new_imageData = struct('image', cell(1, imagesNumber), 'XYpixels', cell(1, imagesNumber),...
        'XYmm', cell(1, imagesNumber), 'H', cell(1, imagesNumber),...
        'R', cell(1, imagesNumber), 't', cell(1, imagesNumber),...
        'K', cell(1, imagesNumber), 'P', cell(1, imagesNumber));
    
    for ii=1:imagesNumber
        new_imageData(ii).image = zeros(480, 640);
        new_imageData(ii).XYpixels = zeros(156, 2);
        new_imageData(ii).XYmm = zeros(156, 2);
        new_imageData(ii).H = zeros(3, 3);
        new_imageData(ii).R = zeros(3, 3);
        new_imageData(ii).t = zeros(3, 1);
        new_imageData(ii).K = zeros(3, 3);
        new_imageData(ii).P = zeros(3, 4); 
    end
    
    % first compute homography H
    
    for ii=1:imagesNumber
        XYpixels = imageData(ii).XYpixels;
        XYmm = imageData(ii).XYmm;
        new_imageData(ii).XYpixels = XYpixels;
        new_imageData(ii).XYmm = XYmm;
        
        A = zeros(2 * length(XYpixels), 9);
        %    b = [];

        for jj=1:length(XYpixels)

            Xpixels = XYpixels(jj, 1);
            Ypixels = XYpixels(jj, 2);
            Xmm = XYmm(jj, 1);
            Ymm = XYmm(jj, 2);

            m = [Xmm; Ymm; 1];
            zero = [0; 0; 0];
            A(jj, :) = [m' zero' -Xpixels*m'];
            A(jj + 1, :) = [zero' m' -Ypixels*m'];
            %        b = [b; 0; 0];

        end

        [~, ~, V] = svd(A);
        h = V(:, end);

        new_imageData(ii).H = reshape(h, [3 3])';

    end
    
    % now find K, R and t (intrinsic and extrinsic parameters)
    
    V = zeros(2 * imagesNumber, 6);
    
    for ii=1:imagesNumber
        currentH = imageData(ii).H;
        
        V(ii, :) = compute_v_ij(1, 2, currentH)';
        V(ii + 1, :) = (compute_v_ij(1, 1,  currentH) - compute_v_ij(2, 2, currentH))';
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
        [U, ~, V] = svd(R);
        R_orthogonal = U * V';
        
        new_imageData(ii).R = R;
        new_imageData(ii).R_orthogonal = R_orthogonal;
        new_imageData(ii).t = lambda * K \ currentH(:, 3);
        new_imageData(ii).K = K; % same for all images
        
        % finally compute matrix P
        new_imageData(ii).P = K * [new_imageData(ii).R, new_imageData(ii).t];
    end
end