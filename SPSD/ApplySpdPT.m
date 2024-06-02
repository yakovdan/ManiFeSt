function Point1PT = ApplySpdPT(PP1, PP2, P1, P2)

    if (nargin < 3) || isempty(P1)
        P1 = SpdMean(PP1);
    end
    if nargin < 4
        P2 = SpdMean(PP2);
    end    

    E = sqrtm(P2 / P1);
    
    N           = length(PP1);
    Point1PT{N} = [];
    for ii = 1 : N
        mXi          = PP1{ii};
        Point1PT{ii} = E * mXi * E';
    end
    
end