import csv
edgeFile = 'edges.csv'
with open(edgeFile, newline='') as csvfile:
    edges = list(csv.DictReader(csvfile))

def bfs(start, end):
    # Begin your code (Part 1)
    visited = set()
    queue = [str(start)]
    parent = {}
    distance = {}
    count = 0
    dist = 0
    visited.add(str(start))
    found = False

    while queue:
        current = queue.pop(0)
        count += 1
        # visited.add(current)
        if current == str(end):
            found = True
            break
        for edge in edges:
            # prevent loop
            if edge['start'] == current and edge['end'] not in visited:
                visited.add(edge['end'])
                parent[edge['end']] = current # used to track the path
                distance[edge['end']] = edge['distance']
                queue.append(edge['end'])
    
    if not found:
        # Return an empty path if no route exists.
        return [], 0, count

    # Reconstruct the path from end to start.
    path = []
    current = str(end)
    while current != str(start):
        path.append(int(current))
        dist += float(distance[current])
        current = parent[current]
    path.append(start)
    path.reverse()
    return path, round(dist, 3), count
    # End your code (Part 1)

if __name__ == '__main__':
    path, dist, num_visited = bfs(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
