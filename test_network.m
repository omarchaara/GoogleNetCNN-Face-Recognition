function test_network(net, image)
    I = imread(image);
    G = imresize(I, [224, 224]);
    
    % Classification de l'image
    [label, prob] = classify(net, G);
    
    % Vérification si le score de probabilité est inférieur à 0,8
    if max(prob) < 0.8
        % Afficher l'image avec le texte comme saut de ligne dans le titre
        figure;
        imshow(G);
        title(sprintf('Personne non détectée\nConfiance : %.2f', max(prob)));
    else
        % Affichage de l'image et du label avec le score de probabilité
        figure;
        imshow(G);
        title({char(label), num2str(max(prob), 2)});
    end
end


