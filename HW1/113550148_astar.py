import csv
edgeFile = 'edges.csv'
with open(edgeFile, newline='') as csvfile:
    edges = list(csv.DictReader(csvfile))
heuristicFile = 'heuristic_values.csv'
with open(heuristicFile, newline='') as csvfile:
    heuristics = list(csv.DictReader(csvfile))


def astar(start, end):
    # Begin your code (Part 4)
    pq = [[str(start), 0, 0]]
    cost_so_far = {str(start): 0}
    totalcost_so_far = {}
    for heuristic in heuristics:
        if heuristic['node'] == str(start):
            totalcost_so_far[str(start)] = float(heuristic[str(end)])
            break
    parent = {}
    distance = {}
    count = 0
    dist = 0
    
    while pq:
        current_node, current_cost, total_cost = pq.pop(0)
        count += 1
        # if current_node in visited:
        #     continue
        if current_node == str(end):  # compare node part only
            break

        for edge in edges:
            if edge['start'] == current_node:
                
                dist_to_end = 0
                for heuristic in heuristics:
                    if heuristic['node'] == edge['end']:
                        dist_to_end = float(heuristic[str(end)])
                        break
                # if first meet the node, then calculate the cost so far
                if edge['end'] not in cost_so_far:
                    cost_so_far[edge['end']] = current_cost + float(edge['distance'])
                    totalcost_so_far[edge['end']] = cost_so_far[edge['end']] + dist_to_end
                    parent[edge['end']] = current_node
                    pq.append([edge['end'], cost_so_far[edge['end']], totalcost_so_far[edge['end']]])

                # if we have already met the node, and we now have shorter path, then update it
                elif totalcost_so_far[edge['end']] > current_cost+float(edge['distance'])+dist_to_end:
                    cost_so_far[edge['end']] = current_cost + float(edge['distance'])
                    totalcost_so_far[edge['end']] = current_cost+float(edge['distance'])+dist_to_end
                    parent[edge['end']] = current_node
                    distance[edge['end']] = edge['distance']
                    pq.append([edge['end'], cost_so_far[edge['end']], totalcost_so_far[edge['end']]])

                # otherwise, we don't need to update anything cause we care only about the shortest path
                
                
                
        # make sure i can get the shortest path every time
        pq = sorted(pq, key=lambda x: x[2])

    # Reconstruct path.
    path = []
    current = str(end)
    while current != str(start):
        path.append(int(current))
        current = parent[current]
    path.append(start)
    path.reverse()
    

    return path, round(cost_so_far[str(end)], 3), count
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
