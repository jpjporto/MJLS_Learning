d = 100;
k = 20;
m = 100;

A = zeros(d,d,m);
B = zeros(d,k,m);
for i = 1:m
    while (1)
        sys = drss(d,1,k);
        
        % Check if all e-values are inside unit circle (drss sometimes creates systems with e-values = 1)
        if (all(abs(eig(sys.a)) < 1))
            break;
        end
    end
    A(:,:,i) = sys.a;
    B(:,:,i) = sys.b;
end

prob = randdirichlet((m-1)*eye(m)+1)';
