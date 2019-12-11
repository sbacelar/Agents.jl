using Agents

mutable struct SchellingAgent <: AbstractAgent
  id::Int # The identifier number of the agent
  pos::Tuple{Int,Int} # The x, y location of the agent
  mood::Bool # whether the agent is happy in its node. (true = happy)
  group::Int # The group of the agent,
             # determines mood as it interacts with neighbors
end

space = Space((10,10), moore = true)

properties = Dict(:min_to_be_happy => 3)
schelling = ABM(SchellingAgent, space; properties = properties)

function initialize(;numagents=320, griddims=(20, 20), min_to_be_happy=3)
    space = Space(griddims, moore = true) # make a Moore grid
    properties = Dict(:min_to_be_happy => 3)
    model = ABM(SchellingAgent, space; properties=properties,
     scheduler = random_activation)
    # populate the model with agents, adding equal amount of the two types of agents
    # at random positions in the model
    for n in 1:numagents
        agent = SchellingAgent(n, (1,1), false, n < numagents/2 ? 1 : 2)
        add_agent_single!(agent, model)
    end
    return model
end

function agent_step!(agent, model)
    agent.mood == true && return # do nothing if already happy
    minhappy = model.properties[:min_to_be_happy]
    neighbor_cells = node_neighbors(agent, model)
    count_neighbors_same_group = 0
    # For each neighbor, get group and compare to current agent's group
    # and increment count_neighbors_same_group as appropriately.
    for neighbor_cell in neighbor_cells
        node_contents = get_node_contents(neighbor_cell, model)
        # Skip iteration if the node is empty.
        length(node_contents) == 0 && continue
        # Otherwise, get the first agent in the node...
        agent_id = node_contents[1]
        # ...and increment count_neighbors_same_group if the neighbor's group is
        # the same.
        neighbor_agent_group = model.agents[agent_id].group
        if neighbor_agent_group == agent.group
            count_neighbors_same_group += 1
        end
    end
    # After counting the neighbors, decide whether or not to move the agent.
    # If count_neighbors_same_group is at least the min_to_be_happy, set the
    # mood to true. Otherwise, move the agent to a random node.
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        move_agent_single!(agent, model)
    end
    return
end

# initialize the model with 370 agents on a 20 by 20 grid.
model = initialize()
step!(model, agent_step!)     # run the model one step
step!(model, agent_step!, 3)  # run the model 3 steps.

model = initialize()
# An array of Symbols for the agent fields that are to be collected.
properties = [:pos, :mood, :group]
# Specifies at which steps data should be collected.
n = 5  # number of time steps to run the simulation
when = 1:n  # At which steps to collect data
# Use the step function to run the model and collect data into a DataFrame.
data = step!(model, agent_step!, n, properties, when=when)
data[1:10, :] # print only a few rows

model = initialize(numagents=370, griddims=(20,20), min_to_be_happy=3);
properties = Dict(:mood => [sum])
n = 5; when = 1:n
data = step!(model, agent_step!, 5, properties, when=when)

# AGAIN
model = initialize()
# An array of Symbols for the agent fields that are to be collected.
properties = [:pos, :mood, :group]
# Specifies at which steps data should be collected.
n = 5  # number of time steps to run the simulation
when = 1:n  # At which steps to collect data
# Use the step function to run the model and collect data into a DataFrame.
data = step!(model, agent_step!, n, properties, when=when)
data[1:10, :] # print only a few rows

# Use the plot2D function from AgentsPlots.jl to plot distribution of agents at any step.
using AgentsPlots

#p = plot2D(data, :group, t=1, nodesize=10)
#p = plot2D(data, :group, t=2, nodesize=10)

for i in 1:2
  p = plot2D(data, :group, t=i, nodesize=10)
end

using DataVoyager
v = Voyager(data)

model = initialize(numagents=370, griddims=(20,20), min_to_be_happy=3);
data = step!(model, agent_step!, 5, properties, when=when, replicates=5)

using Distributed
addprocs(2)

data = step!(model, agent_step!, 2, properties,
             when=when, replicates=5, parallel=true)

paramscan

happyperc(moods) = count(x -> x == true, moods)/length(moods)

properties= Dict(:mood=>[happyperc])
parameters = Dict(:min_to_be_happy=>collect(2:5), :numagents=>[200,300], :griddims=>(20,20))

data = paramscan(parameters, initialize;
       properties=properties, n = 3, agent_step! = agent_step!)

data = paramscan(parameters, initialize;
       properties=properties, n = 3, agent_step! = agent_step!,
       replicates=3)

using DataFrames: Not, select!
using Statistics: mean
data_mean = Agents.aggregate(data, [:step, :min_to_be_happy, :numagents],  mean);
select!(data_mean, Not(:replicate_mean))
