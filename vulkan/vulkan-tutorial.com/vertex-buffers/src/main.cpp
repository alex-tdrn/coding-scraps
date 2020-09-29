#include "VertexBuffers.h"

#include <iostream>

int main()
{
	VertexBuffers app;

	try
	{
		app.run();
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
