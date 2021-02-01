function zhang_estimation(imageData, imagesNumber)
    
    % code copied from project1
    % XYmm and XYpixels must be previously evaluated
    
    % first compute homography H
    
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

        [~, ~, V] = svd(A);
        h = V(:, end);

        imageData(ii).H = reshape(h, [3 3])';

    end
    
    % now find K, R and t (intrinsic and extrinsic parameters)
    
    V = [];
    
    for ii=1:imagesNumber
        currentH = imageData(ii).H;
        
        V = [V; compute_v_ij(1, 2, currentH)';...
            (compute_v_ij(1, 1,  currentH) - compute_v_ij(2, 2, currentH))'];
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
        
        imageData(ii).R = R;
        imageData(ii).R_orthogonal = R_orthogonal;
        imageData(ii).t = lambda * K \ currentH(:, 3);
        
        % finally compute matrix P
        imageData(ii).P = K * [imageData(ii).R, imageData(ii).t];
    end


end