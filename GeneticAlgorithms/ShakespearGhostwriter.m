clear
close all

%define target
target_string = 'sein oder nicht sein das ist die frage';
tarlen = length(target_string);

%Kapital case letter alphabet begins at A = char(65) till Z = char(90)
%Lower case letter alphabet begins at  a = char(97) till z = char(122)
%char(32) is the spacebar

%parameters for genetic algorithm
population = 200;
Generations = 250;
parents = char(zeros(2,tarlen));
mutation_rate = 0.02;
string_pop = char(zeros(population,tarlen));
string_fitness = zeros(population,1);
lowercase_char_pool = char([32,97:122]);
uppercase_char_pool = char([32,65:90]);
averageFitness = zeros(1,Generations);
maxFitness = averageFitness;
fig = figure;
fig.Position = [1000,600,600,600];
ax = axes(fig);


%create population of random strings of target length
for e = 1:population
    for i = 1:tarlen
        string_pop(e,i) = randsample(lowercase_char_pool,1);
    end
end

for k = 1:Generations
    %calculate fitness
    for e = 1:population
        string_fitness(e) = sum(string_pop(e,:)==target_string);
    end
    string_fitness = string_fitness./tarlen * 100; %normalize, not really necessary

    %plot results in command window
    idefix = find(string_fitness==max(string_fitness));
    string_pop(idefix(1),:)
    currentBestFitness = max(string_fitness);
    maxFitness(k) = currentBestFitness;
    averageFitness(k) = mean(string_fitness);
    plot(1:k,nonzeros(averageFitness),'LineWidth',2,'color','r')
    axis(ax, [0.99, k, 0, 100])
    hold on
    pause(10e-10)
    if currentBestFitness==100
        hold off
        newplot
        plot(1:k,nonzeros(maxFitness),1:k,nonzeros(averageFitness),'LineWidth',2)
        title('Evolution Report')
        ylabel(ax, 'Fitness [%]')
        xlabel(ax, 'Generation')
        legend(ax, ['max';'avg'],'Location','SE')
        colororder(ax, ['r';'b'])
        axis(ax, [0.99, k, 0, 100])
        finishingGeneration = k
        break
    end

%     %pick parents
%     string_fitnessS = string_fitness; %copy array for independent manipulation
%     probabiltyFit = string_fitnessS/sum(string_fitnessS);
%     choice = rand(1);
% 
%     for r = 1:size(parents,1)
%         for e = 1:population
%             if e==1 && probabiltyFit(1) > choice
%                 parents(r,:) = string_pop(e,:);
%                 string_fitnessS(e) = 0; %remove the picked one from the the list of possible strings by setting its fitness to zero
%                 probabiltyFit = string_fitnessS/sum(string_fitnessS); %update probabilities
%                 break
%             end
%             if sum(probabiltyFit(1:e-1)) < choice && choice < sum(probabiltyFit(1:e))
%                 parents(r,:) = string_pop(e,:);
%                 string_fitnessS(e) = 0; %remove the picked one from the the list of possible strings by setting its fitness to zero
%                 probabiltyFit = string_fitnessS/sum(string_fitnessS); %update probabilities
%                 break
%             end
%         end
%         choice = rand(1); %spin again the wheel of choice
%     end
  
    fitnessCopy = string_fitness; %copy to not damage anything
    for r = 1:size(parents,1)
        [value,index] = max(fitnessCopy);
        parents(r,:) = string_pop(index,:);
        fitnessCopy(fitnessCopy==value) = 0;  %take the fittest then the second fittest thend the third fittest and so on
    end

    %mate parents and generate offspring
    for e = 1:population
        randSep = randi(tarlen); %do crossing over of random size
        string_pop(e,:) = [parents(1,(1:randSep)),parents(2,(randSep+1:end))];
    end

    %add mutations
    for e = 1:population
        for i = 1:tarlen
            %conscious mutagenesis^^
            logicalMutation = (string_pop(e,:)==target_string); %create a logical array to check if a given position in the string is correct or not. Then only mutate if the given letter is not yet correct. This allows for much higher mutation rates since it now affects only the faulty chars thus giving higher proability of make it more correct than more incorrect
            logicalMutation = zeros(1,tarlen);
            if logicalMutation(i)==0 && mutation_rate > rand(1)
                string_pop(e,i) = randsample(lowercase_char_pool,1);
            end
        end
    end
end