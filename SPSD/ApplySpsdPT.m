function [AaPT, GAaPT, PAaPT] = ApplySpsdPT(AA, BB, r)

    Symm  = @(M) (M + M') / 2;
    N1    = length(AA);

    [C1, G1, P1] = SpsdMean(AA, r);
    [C2, G2, P2] = SpsdMean(BB, r);

    [O1, ~, O2] = svd(G1' * G2);
    G2          = G2  * O2 * O1';
    P2          = G2' * C2 * G2;
    
    PAA{N1} = [];
    GAA{N1} = [];
    for ii = 1 : N1
        Ci      = Symm(AA{ii});
        [Gi, ~] = eigs(Ci, r);
        
        [Oi, ~, OWi] = svd(Gi' * G1);
        Gi           = Gi  * Oi * OWi';
        Ti           = Gi' * Ci * Gi;
        GAA{ii}      = Gi;
        PAA{ii}      = Symm(Ti);
    end
    
    PAaPT = ApplySpdPT     (PAA, [], P1, P2);
    GAaPT = ApplyGrassmanPT(GAA, [], G1, G2);
    
    AaPT{N1} = [];
    for ii = 1 : N1
        mGAPTi   = GAaPT{ii};
        mPAPTi   = PAaPT{ii};
        AaPT{ii} = mGAPTi * mPAPTi * mGAPTi';
    end
end
