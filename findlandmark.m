function landmark = findlandmark(I)
% finds the topmost rightmost point where the right ventricle connects to the left ventricle    
    

    %ACDC data%%%%%%%%%%%%%%%%
    %remove left ventricle
    I(find(I == 3)) = 2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %UKBB data
    %I(find(I == 1)) = 2;
    
    %slide 2x2 window, look if it contains all tree labels 
    corners = [];
    [m,n] = size(I);
    for i = 1:m-1
        for j = 1:m-1
            patch = I(i:i+1, j:j+1);
            if length(unique(patch)) == 3
                corners = [corners; [i,j]];
            end  
        end
    end
    
    if length(corners) ~= 2
        sprintf('%d corners detected', length(corners))
    end
    
    if (length(corners) == 0)
        landmark = [-1, -1];
    else
        landmark = [];
    
        [topcorner, topid] = min(corners(:,1));

        if length(find(corners(:,1)==topcorner)) > 1
            topcorners = corners(corners(:,1)==topcorner,:);
            [rightcorner, rightid] = max(topcorners(:,2));
            landmark = topcorners(rightid, :);
        else
            landmark = corners(topid,:);
        end
    end
    
end