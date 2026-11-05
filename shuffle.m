function v=shuffle(v)
%Shuffles the elemetns in a row or column but it cant handle matrices
     v=v(randperm(length(v)));
 end