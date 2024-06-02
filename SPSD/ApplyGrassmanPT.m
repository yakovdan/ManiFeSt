function GG1PT = ApplyGrassmanPT2(GG1, GG2, G1, G2)
    
    N1     = length(GG1);
    [d, r] = size(GG1{1});
    Grass  = grassmannfactory(d, r, 1);

    if (nargin < 3) || isempty(G1)
        G1 = GrassmanMean(GG1);
    end
    if nargin < 4
        G2 = GrassmanMean(GG2);
    end
    
    [Q1, ~] = qr(G1);
    G1perp  = Q1(:,r+1:d);
    A       = G1perp' *  Grass.log(G1, G2);
    AA      = [zeros(r), -A';
               A       , zeros(d-r)];
    Q2      = Q1 * expm(AA);
    O       = Q2 * Q1';

    GG1PT{N1} = [];
    for ii = 1 : N1
        GG1PT{ii} = O * GG1{ii};
    end
end
