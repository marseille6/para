#基于cuda实现n体问题的一个简单算法
##基本思想是
1.为每个小球分配一个gpu处理单元，每一帧去遍历与其它n-1个小球的碰撞关系.
2.等待所有小球处理完成后开始下一帧的小球碰撞检测.
