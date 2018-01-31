function Io = L0min(Ii,lambda)
% Ii is the input image, lambda is the smoothing parameter, 

tic
B0 = 2*lambda; % control parameter
Bmax = 1e5; % control parameter
k = 2; % rate parameter

% get dimensions of image
[H, W, C] = size(Ii);

% Fourier transform of the forward difference operator
FDx = psf2otf([1,-1],[H W]);
FDy = psf2otf([1;-1],[H W]);

% convert image to double and change to Fourier domain
S = im2double(Ii);
FI = fft2(S);

% initialize beta
B = B0;

% compute FDx^*FDx + FDy^*FDy outside of loop to maximize efficiency
grad = abs(FDx).^2 + abs(FDy).^2;

% handle RGB images by simply computing each gradient in parallel
if C > 1
    grad = repmat(grad, [1 1 C]);
    FDx = repmat(FDx, [1 1 C]);
    FDy = repmat(FDy, [1 1 C]);
end

while B < Bmax
    % Using S, solve for h and v
    
    % pad S so that the forward difference yields a matrix of same size
    Spadh = [S, S(:,end,:)];
    Spadv = [S; S(1,:,:)];
    h = diff(Spadh,1,2);
    v = diff(Spadv,1,1);
    % change values of h,v to 0 according to minimization rule
    if C == 1
        index = h.^2+v.^2 <= lambda/B;
        h(index) = 0;
        v(index) = 0;
    else
        % in RGB case, simply sum the gradients
        gradsum = sum((h.^2+v.^2),3);
        gradsum = repmat(gradsum,[1 1 C]);
        index = gradsum <= lambda/B;
        h(index) = 0;
        v(index) = 0;
    end
    
    % Using h,v, solve for S
    FS =(FI+B*(conj(FDx).*fft2(h) + conj(FDy).*fft2(v)))./(1+B*grad);
    S = real(ifft2(FS));
    
    % update parameter
    B = B*k;
    fprintf('*');
end
fprintf(' done!\n');
Io = S;
toc
end
