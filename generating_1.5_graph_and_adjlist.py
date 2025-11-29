import networkx as nx
import matplotlib.pyplot as plt

def generate_instance(a: int, c: int = 3) -> nx.Graph:
    """
    Generate a graph with inner and outer cycles with apecific cross-connections that yield a 1.5 integrality gap
    
    Args:
        a: Size of the inner cycle
        c: Cross-connection parameter (default: 3)
    
    Returns:
        A NetworkX graph with the specified structure
    """
    b = 2 * a  # outer cycle size
    G = nx.Graph()
    inner_range = range(a)
    
    # Add inner cycle edges (0..a-1)
    for i in inner_range:
        G.add_edge(i, (i + 1) % a)

    # Add outer cycle edges (a..a+b-1)
    for i in range(b):
        G.add_edge(i + a, ((i + 1) % b) + a) #a is an outer cycle offset

    #debugging
    #print(f"Nodes after adding inner and outer cycles: {G.nodes()}")
    #nx.draw(G, with_labels=True)
    #plt.show()

    #cross connections - each inner node connects to two outer nodes (one odd indexed, one even indexed)
    available = [i for i in range(a,a+b)]
    odd_idx = a+b-1
    for i in inner_range:
        G.add_edge(i, odd_idx)
        print(f"Connecting inner node {i} to outer node {odd_idx}")
        available.remove(odd_idx)
        odd_idx = (odd_idx + c - 1) % b
        if odd_idx < a:
            odd_idx += b
    print("Available after odd connections:", available)

    a_iterator = [a-1]
    for i in range(0, a-1):
        a_iterator.append(i)

    even_idx = available[0]
    for i in a_iterator:
        G.add_edge(i, even_idx)
        print(f"Connecting inner node {i} to outer node {even_idx}")
        print(available)
        available.remove(even_idx)
        even_idx = (even_idx + c - 1) % b
        if even_idx < a:
            even_idx += b

    return G

def parse_adjlist_to_dict(adjlist_generator):
    adj_dict = {}
    for line in adjlist_generator:
        parts = list(map(int, line.strip().split()))
        if not parts:
            continue
        node = parts[0]
        neighbors = parts[1:]
        adj_dict[node] = neighbors
    return adj_dict

if __name__ == "__main__":
    
    #Create graph of inner cycle size a
    G = generate_instance(a=20)
    
    generated_adjlist = nx.generate_adjlist(G)
    result_dict = parse_adjlist_to_dict(generated_adjlist)
    print("Converted Adjacency Dictionary:")
    print(result_dict)

    nx.draw(G, with_labels=True)
    plt.show()

