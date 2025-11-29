# MISR

Instances created in the form of intersection graphs, adjacency lists, and rectangle coordinates (permuted or not).

This is not a smooth pipeline. Most results are printed to console, and the user must copy and paste results across files to get the data. Since there is a lot of experimentation, the code is a bit messy, but it works.

## Produce 1.5 gap intersection graph
- Run the file generating_1.5_graph_and_adjlist.py
- Edit line 74 to change the size of the graph: "G = generate_instance(a=20)"
- Change a (the inner cycle) to be any size of your choice

## Produce corresponding rectangle coordinates
- Copy the adjacency dictionary from the output of the above
- Paste into a new file in the form "etc.adjlist"
- Paste the same dictionary into the file adjlist_to_config.py on line 115 (to be the value for the variable adj_list):
  
    adj_list = {0: [1, 19, 59, 22], ...}
    
- Run adjlist_to_config.py

- Copy the coordinates output of the above
- Paste into a new file in the form "etc.config"
- Run the file permutation_generation.py
- Edit lines 393 and 394 to the names of your new adjlist and config files:
  
    rects = load_rectangles_from_file("configs/20_40.config")
  
    adj = load_adjacency_dict("adjlists/20_40.adjlist")

The result will be a plot of the rectangle coordinates, including the integrality gap ratio, and the output will show the exact coordinate values.
