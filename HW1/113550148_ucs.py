import csv
edgeFile = 'edges.csv'
with open(edgeFile, newline='') as csvfile:
    edges = list(csv.DictReader(csvfile))

def ucs(start, end):
    # Begin your code (Part 3)
    pq = [[str(start), 0]]
    cost_so_far = {str(start): 0}
    parent = {}
    count = 0
    dist = 0
    
    while pq:
        current_node, current_cost = pq.pop(0)
        count += 1
        # if current_node in visited:
        #     continue
        if current_node == str(end):  # compare node part only
            break

        for edge in edges:
            if edge['start'] == current_node:
                # if first meet the node, then calculate the cost so far
                if edge['end'] not in cost_so_far:
                    cost_so_far[edge['end']] = cost_so_far[edge['start']] + float(edge['distance'])
                    parent[edge['end']] = current_node
                    pq.append([edge['end'], cost_so_far[edge['end']]])

                # if we have already met the node, and we now have shorter path, then update it
                elif cost_so_far[edge['end']] > cost_so_far[edge['start']]+ float(edge['distance']):
                    cost_so_far[edge['end']] = cost_so_far[edge['start']] + float(edge['distance'])
                    parent[edge['end']] = current_node
                    pq.append([edge['end'], cost_so_far[edge['end']]])

                # otherwise, we don't need to update anything cause we care only about the shortest path
                
                
                
        # make sure i can get the shortest path every time
        pq = sorted(pq, key=lambda x: x[1])

    # Reconstruct path.
    path = []
    current = str(end)
    while current != str(start):
        path.append(int(current))
        current = parent[current]
    path.append(start)
    path.reverse()

    return path, round(cost_so_far[str(end)], 3), count
    # End your code (Part 3)

if __name__ == '__main__':
    path, dist, num_visited = ucs(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
