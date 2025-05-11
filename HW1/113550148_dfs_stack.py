import csv
edgeFile = 'edges.csv'
with open(edgeFile, newline='') as csvfile:
    edges = list(csv.DictReader(csvfile))


def dfs(start, end):
    # Begin your code (Part 2)
    visited = set()
    stack = [str(start)]
    parent = {}
    distance = {}
    count = 0
    dist = 0

    while stack:
        current = stack.pop(-1)
        count += 1
        visited.add(current)
        if current == str(end):
            break
        for edge in edges:
            if edge['start'] == current and edge['end'] not in visited:
                parent[edge['end']] = current
                distance[edge['end']] = edge['distance']
                stack.append(edge['end'])

    path = []
    current = str(end)
    while current != str(start):
        path.append(int(current))
        dist += float(distance[current])
        current = parent[current]
    path.append(start)
    path.reverse()
    return path, round(dist, 3), count
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
