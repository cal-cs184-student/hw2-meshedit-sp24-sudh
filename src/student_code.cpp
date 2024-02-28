#include "student_code.h"
#include "mutablePriorityQueue.h"
#include <math.h>
using namespace std;

#define _USE_MATH_DEFINES
#define SQRT3

namespace CGL
{

  /**
   * Evaluates one step of the de Casteljau's algorithm using the given points and
   * the scalar parameter t (class member).
   *
   * @param points A vector of points in 2D
   * @return A vector containing intermediate points or the final interpolated vector
   */
  std::vector<Vector2D> BezierCurve::evaluateStep(std::vector<Vector2D> const &points)
  { 
    // TODO Part 1.
    int n = points.size();
    std::vector<Vector2D> to_ret;
    for(int i = 0; i < n - 1; i++) {
      to_ret.push_back((1.0 - t)*points[i] + (t)*points[i+1]);
    }
    return to_ret;
  }

  /**
   * Evaluates one step of the de Casteljau's algorithm using the given points and
   * the scalar parameter t (function parameter).
   *
   * @param points    A vector of points in 3D
   * @param t         Scalar interpolation parameter
   * @return A vector containing intermediate points or the final interpolated vector
   */
  std::vector<Vector3D> BezierPatch::evaluateStep(std::vector<Vector3D> const &points, double t) const
  {
    // TODO Part 2.
    int n = points.size();
    std::vector<Vector3D> to_ret;
    for(int i = 0; i < n - 1; i++) {
      to_ret.push_back((1.0 - t)*points[i] + (t)*points[i+1]);
    }
    return to_ret;
  }

  /**
   * Fully evaluates de Casteljau's algorithm for a vector of points at scalar parameter t
   *
   * @param points    A vector of points in 3D
   * @param t         Scalar interpolation parameter
   * @return Final interpolated vector
   */
  Vector3D BezierPatch::evaluate1D(std::vector<Vector3D> const &points, double t) const
  {
    // TODO Part 2.
    int n = points.size();
    auto next = points;
    for(int i = 1; i < n; i++) {
      next = evaluateStep(next,t);
    }

    return next[0];
  }

  /**
   * Evaluates the Bezier patch at parameter (u, v)
   *
   * @param u         Scalar interpolation parameter
   * @param v         Scalar interpolation parameter (along the other axis)
   * @return Final interpolated vector
   */
  Vector3D BezierPatch::evaluate(double u, double v) const 
  {  
    // TODO Part 2.
    int n = controlPoints.size();
    std::vector<Vector3D> oth_pts;
    for(int i = 0; i < n; i++){
      oth_pts.push_back(evaluate1D(controlPoints[i], u));
    }
    return evaluate1D(oth_pts,v);
  }

  Vector3D Vertex::normal( void ) const
  {
    // TODO Part 3.
    // Returns an approximate unit normal at this vertex, computed by
    // taking the area-weighted average of the normals of neighboring
    // triangles, then normalizing.
    Vector3D totalArea( 0., 0., 0.);
    HalfedgeCIter h = halfedge();
    Vector3D posA;
     do
     {
        {
          posA = h->vertex()->position;
          Vector3D pi = posA - position;
          Vector3D pj = h->next()->vertex()->position - posA;
           totalArea = totalArea +  0.5*cross( pi, pj );
        }
       

      h = h->next();
     }
     while( h != halfedge() );
           
    return totalArea.unit();
  }

  EdgeIter HalfedgeMesh::flipEdge( EdgeIter e0 )
  {
    // TODO Part 4.
    // This method should flip the given edge and return an iterator to the flipped edge.


    //First collect all halfedges
    HalfedgeIter h0 = e0->halfedge();
    HalfedgeIter h1 = h0->next();
    HalfedgeIter h2 = h1->next();
    HalfedgeIter h3 = h0->twin();
    HalfedgeIter h4 = h3->next();
    HalfedgeIter h5 = h4->next();


    //now all edges
    EdgeIter e1 = h1->edge();
    EdgeIter e2 = h2->edge();
    EdgeIter e3 = h4->edge();
    EdgeIter e4 = h5->edge();

    //vertices
    VertexIter b = h0->vertex();
    VertexIter c = h3->vertex();
    VertexIter a = h2->vertex();
    VertexIter d = h5->vertex();
    

    //faces
    FaceIter f0 = h0->face();
    FaceIter f1 = h3->face();

    //first lets change the half edges
    h0->setNeighbors(h2,h3,d,e0,f0);
    h1->setNeighbors(h3,h1->twin(),c,e1,f1);
    h2->setNeighbors(h4,h2->twin(),a,e2,f0);
    h3->setNeighbors(h5,h0,a,e0,f1);
    h4->setNeighbors(h0,h4->twin(),b,e3,f0);
    h5->setNeighbors(h1,h5->twin(),d,e4,f1);

    //assign edges
    e0->halfedge() = h0;
    e1->halfedge() = h1;
    e2->halfedge() = h2;
    e3->halfedge() = h4;
    e4->halfedge() = h5;
    //assign vertices
    b->halfedge() = h4;
    c->halfedge() = h1;
    a->halfedge() = h3;
    d->halfedge() = h0;

    //finally faces
    f0->halfedge() = h0;
    f1->halfedge() = h3;



    
  
   






    
  
    
  return e0;

   
   

  }

  VertexIter HalfedgeMesh::splitEdge(EdgeIter e0)
{
    if (e0->isBoundary()) {
			HalfedgeIter h0 = e0->halfedge();
      if(h0->next()->next()->vertex()->isBoundary()) h0 = h0->twin();
      HalfedgeIter h1 = h0->next();
      HalfedgeIter h2 = h1->next();

      //now get the edges, vertices and face

      FaceIter f0 = h0->face();
      EdgeIter e0 = h0->edge();
      EdgeIter e1 = h1->edge();
      EdgeIter e2 = h2->edge();
      VertexIter v0 = h0->vertex();
      VertexIter v1 = h1->vertex();
      VertexIter v2 = h2->vertex();

      //now time to ad the new edge - let's create the points and all first
      VertexIter v3 = newVertex();
      v3->position = 0.5*(v0->position + v1->position);
      EdgeIter e3 = newEdge();
       EdgeIter e4 = newEdge();
      FaceIter f1 = newFace();

      HalfedgeIter h3 = newHalfedge();
      HalfedgeIter h4 = newHalfedge();
      HalfedgeIter h5 = newHalfedge();
      HalfedgeIter h6 = newHalfedge();



      //let the reassignment begin
      h0->twin()->setNeighbors(h6,h0,v1,e0, h0->twin()->face());
      h0->setNeighbors(h1, h0->twin(),v3, e0,f0);
      h1->setNeighbors(h3, h1->twin(), v1, e1, f0);
      h2->setNeighbors(h5, h2->twin(), v2, e2, f1);
      h3->setNeighbors(h0,h4,v2,e3,f0);
      h4->setNeighbors(h2,h3,v3,e3,f1);
      h5->setNeighbors(h4, h6, v0, e4, f1);
      h6->setNeighbors(h0->twin()->next(),h5,v3,e4, h0->twin()->face());

      //reasign edges
      e0->halfedge() = h0;
      e1->halfedge() = h1;
      e2->halfedge() = h2;
      e3->halfedge() = h4;
      e4->halfedge() = h5;

      //reassign vertces
      v0->halfedge() = h5;
      v1->halfedge() = h1;
      v2->halfedge() = h2;
      v3->halfedge() = h0;

      //faces
      f0->halfedge() = h0;
      f1->halfedge() = h5;

      return v3;



		}

		// collect halfedges
		HalfedgeIter h0 = e0->halfedge();
		HalfedgeIter h1 = h0->next();
		HalfedgeIter h2 = h1->next();
		HalfedgeIter h3 = h0->twin();
		HalfedgeIter h4 = h3->next();
		HalfedgeIter h5 = h4->next();
		HalfedgeIter h6 = h1->twin();
		HalfedgeIter h7 = h2->twin();
		HalfedgeIter h8 = h4->twin();
		HalfedgeIter h9 = h5->twin();
		// collect vertices
		VertexIter v0 = h0->vertex();
		VertexIter v1 = h3->vertex();
		VertexIter v2 = h2->vertex();
		VertexIter v3 = h5->vertex();
		// collect edges
		EdgeIter e1 = h1->edge();
		EdgeIter e2 = h2->edge();
		EdgeIter e3 = h4->edge();
		EdgeIter e4 = h5->edge();
		// collect faces
		FaceIter f0 = h0->face();
		FaceIter f1 = h3->face();

		// create mew halfedges
		HalfedgeIter h10 = newHalfedge();
		HalfedgeIter h11 = newHalfedge();
		HalfedgeIter h12 = newHalfedge();
		HalfedgeIter h13 = newHalfedge();
		HalfedgeIter h14 = newHalfedge();
		HalfedgeIter h15 = newHalfedge();
		// create new vertices
		VertexIter v4 = newVertex();
		v4->position = 0.5*(v0->position + v1->position);
		// create new edges
		EdgeIter e5 = newEdge();
		EdgeIter e6 = newEdge();
		EdgeIter e7 = newEdge();
		// create new faces
		FaceIter f2 = newFace();
		FaceIter f3 = newFace();

		// assign halfedges
	
    h0->setNeighbors(h1,h3,v4,e0,f0);
    h1->setNeighbors(h12,h6,v1,e1,f0);
    h2->setNeighbors(h15,h7,v2,e2,f3);
    h3->setNeighbors(h10,h0,v1,e0,f1);
    h4->setNeighbors(h11,h8,v0,e3,f2);
		h5->setNeighbors(h3,h9,v3,e4,f1);
    h6->setNeighbors(h6->next(),h1,v2,e1,h6->face());
    h7->setNeighbors(h7->next(),h2,v0,e2,h7->face());
    h8->setNeighbors(h8->next(),h4,v3,e3,h8->face());
		h9->setNeighbors(h9->next(),h5,v1,e4,h9->face());
    h10->setNeighbors(h5,h11,v4,e6,f1);
    h11->setNeighbors(h14,h10,v3,e6,f2);
		h12->setNeighbors(h0,h13,v2,e7,f0);
		h13->setNeighbors(h2,h12,v4,e7,f3);
    h14->setNeighbors(h4,h15,v4,e5,f2);
    h15->setNeighbors(h13,h14,v0,e5,f3);

		// ertices
		v0->halfedge() = h15;
		v1->halfedge() = h3;
		v2->halfedge() = h12;
		v3->halfedge() = h11;
		v4->halfedge() = h0;

		//edges
		e0->halfedge() = h0;
		e1->halfedge() = h1;
		e2->halfedge() = h2;
		e3->halfedge() = h4;
		e4->halfedge() = h5;
		e5->halfedge() = h14;
		e6->halfedge() = h10;
		e7->halfedge() = h13;

		//faces
		f0->halfedge() = h0;
		f1->halfedge() = h3;
		f2->halfedge() = h14;
		f3->halfedge() = h15;

		return v4;
}


 VertexIter HalfedgeMesh::AddCentroid(FaceIter f0) {

  HalfedgeIter h0 = f0->halfedge();
  HalfedgeIter h1 = h0->next();
  HalfedgeIter h2 = h1->next();
  VertexIter v0 = h0->vertex();
  VertexIter v1 = h1->vertex();
  VertexIter v2 = h2->vertex();

  VertexIter v3 = newVertex();

	v3->position = (v1->position + v2->position + v0->position)/3.0;
  
  EdgeIter e0 = h0->edge();
  EdgeIter e1 = h1->edge();
  EdgeIter e2 = h2->edge();

  FaceIter f1 = newFace();
	FaceIter f2 = newFace();

  HalfedgeIter h3 = newHalfedge();
  HalfedgeIter h4 = newHalfedge();
  HalfedgeIter h5 = newHalfedge();
  HalfedgeIter h6 = newHalfedge();
  HalfedgeIter h7 = newHalfedge();
  HalfedgeIter h8 = newHalfedge();

  EdgeIter e3 = newEdge();
  EdgeIter e4 = newEdge();
  EdgeIter e5 = newEdge();

  //set halfedges
  h0->setNeighbors(h3, h0->twin(),v0, e0,f0);
  h1->setNeighbors(h5, h1->twin(),v1, e1,f1);
  h2->setNeighbors(h7, h2->twin(),v2, e2,f2);
  h3->setNeighbors(h4,h6,v1,e3,f0);
  h4->setNeighbors(h0,h7,v3,e5,f0);
  h5->setNeighbors(h6,h8,v2,e4,f1);
  h6->setNeighbors(h1,h3,v3,e3,f1);
  h7->setNeighbors(h8,h4,v0,e5,f2);
  h8->setNeighbors(h2,h5,v3,e4,f2);

  //set edges
  e0->halfedge() = h0;
  e1->halfedge() = h1;
  e2->halfedge() = h2;
  e3->halfedge() = h3;
  e5->halfedge() = h4;
  e4->halfedge() = h7;

  //set faces
  f0->halfedge() = h0;
  f1->halfedge() = h1;
  f2->halfedge() = h2;

  //set vertices
  v3->halfedge() = h4;
  v0->halfedge() = h0;
  v1->halfedge() = h1;
  v2->halfedge() = h2;


  return v3;



 }




  void MeshResampler::upsample( HalfedgeMesh& mesh )
  {
    // TODO Part 6.
    // This routine should increase the number of triangles in the mesh using Loop subdivision.
    // One possible solution is to break up the method as listed below.

    // 1. Compute new positions for all the vertices in the input mesh, using the Loop subdivision rule,
    // and store them in Vertex::newPosition. At this point, we also want to mark each vertex as being
    // a vertex of the original mesh.
    
    // 2. Compute the updated vertex positions associated with edges, and store it in Edge::newPosition.
    
    // 3. Split every edge in the mesh, in any order. For future reference, we're also going to store some
    // information about which subdivide edges come from splitting an edge in the original mesh, and which edges
    // are new, by setting the flat Edge::isNew. Note that in this loop, we only want to iterate over edges of
    // the original mesh---otherwise, we'll end up splitting edges that we just split (and the loop will never end!)
    
    // 4. Flip any new edge that connects an old and new vertex.

    // 5. Copy the new vertex positions into final Vertex::position.

  

    #ifdef SQRT3
    //sqrt 3 subdivision first introduces a 4th vertex in each triangle. Then we flip all the original edges
    //first let's create a function to add the centroid - this is the function AddCentroid, right above
    //first set all faces to be old
    for(FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); ++f){
      f->isNew = false;
    }

    //now follow the same steps recommended for the bisection splitting -- lets start with assigning new positions to existing vertices
    for (VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v) {
			int n = v->degree();
      //following the paper, alpha = (4 - cos(2pi/n))/9
			double alpha = (4.0 - 2.0*cos(2.0*M_PI/static_cast<double>(n)))/9.0;
			 Vector3D sum = 0; 
        HalfedgeIter h = v->halfedge();
        HalfedgeIter h_init = h;
        v->newPosition = (1.0 - alpha) * v->position;
        do {
          VertexIter v = h->twin()->vertex();
          sum += v->position;
          h = h->twin()->next();
        } while (h != h_init);

      
			 v->newPosition = v->newPosition + (alpha/static_cast<double>(n))*sum;
			v->isNew = false;
		}

      //next the edges
      for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); ++e) {
        HalfedgeIter h = e->halfedge();
			VertexIter v0= h->vertex();
			  VertexIter v1 = h->twin()->vertex();
        VertexIter v2 = h->next()->next()->vertex();
			  VertexIter v3 = h->twin()->next()->next()->vertex();
			  e->newPosition = 0.5* (v0->position + v1->position) ;
		}

    //now call addCentroid on each face, and mark the new edges and vertices
    std::vector<EdgeIter> old_edges;
    for(FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); ++f) {
      if (!f->isNew) {
      VertexIter v = mesh.AddCentroid(f);
     

      HalfedgeIter h = v->halfedge();
       v->position = (1.0/3.0)*(h->next()->vertex()->newPosition + h->next()->next()->vertex()->newPosition + h->twin()->next()->next()->vertex()->newPosition);
       v->isNew = true;
      EdgeIter e0 = h->next()->edge();
      EdgeIter e1 = h->next()->next()->twin()->next()->edge();   
      EdgeIter e2 = h->twin()->next()->next()->edge();
      EdgeIter e3 = h->edge();
      EdgeIter e4 = h->next()->next()->edge();
      EdgeIter e5 = h->twin()->next()->edge();
      e3->isNew = true;
      e4->isNew = true;
      e5->isNew = true;
      //now set faces to be new
      FaceIter f0 = h->face();
      FaceIter f1 = h->twin()->face();
      FaceIter f2 = h->next()->next()->twin()->face();
      f0->isNew = true;
      f1->isNew = true;
      f2->isNew = true;
      old_edges.push_back(e0);
      old_edges.push_back(e1);
      old_edges.push_back(e2);
      }
  

    }

    for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); ++e) {
      if(!e->isNew)
			mesh.flipEdge(e);
		}

		for (VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v) {
			if (!v->isNew) v->position = v->newPosition;
		}





    //Then flip the original edges

    #else 
      for (VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v) {
			int n = v->degree();
      //need to add one for boundray edges
      if(v->isBoundary()) {
        n = n + 1;
        v->newPosition = (3.0/4.0)*v->position;
        v->newPosition = v->newPosition + (1.0/8.0)*v->halfedge()->twin()->vertex()->position;
        v->newPosition = v->newPosition + (1.0/8.0)*v->halfedge()->twin()->next()->twin()->vertex()->position;

      } else {
        double u = n == 3 ?  3.0/ 16.0 : 3.0/(8.0 * static_cast<double>(n));
        Vector3D sum = 0; 
        HalfedgeIter h = v->halfedge();
        HalfedgeIter h_init = h;
        do {
          VertexIter v = h->twin()->vertex();
          sum += v->position;
          h = h->twin()->next();
        } while (h != h_init);
        v->newPosition = (1.0 - static_cast<double>(n)*u) * v->position + u*sum;
        
      }
      v->isNew = false;
		}

		for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); ++e) {
			HalfedgeIter h = e->halfedge();
			
      if(e->isBoundary()) {
        VertexIter v0= h->vertex();
			  VertexIter v1 = h->twin()->vertex();
        e->newPosition = 0.5*(v0->position + v1->position);
      } else {
        VertexIter v0= h->vertex();
			  VertexIter v1 = h->twin()->vertex();
        VertexIter v2 = h->next()->next()->vertex();
			  VertexIter v3 = h->twin()->next()->next()->vertex();
			  e->newPosition = (double) 3/ (double) 8 * (v0->position + v1->position) + (double) 1/ (double) 8 * (v2->position+ v3->position);
      }
			e->isNew = false;
		}

		//split and then flip. We'll store all the new edges in an array
		std::vector<EdgeIter> new_edges;
		for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); ++e) {
			if (!e->isNew) {
        if(!e->isBoundary()){
          VertexIter v = mesh.splitEdge(e);
          v->position = e->newPosition;
          v->isNew = true;
          HalfedgeIter h = v->halfedge();
          EdgeIter e0 = h->edge();
          EdgeIter e1 = h->twin()->next()->twin()->next()->edge();   //e0,e1 are the edge that we added the vertex on, so they are not really new 
          EdgeIter e2 = h->next()->next()->edge();
          EdgeIter e3 = h->twin()->next()->edge();
          e0->isNew = true;
          e1->isNew = true;
          e2->isNew = true;
          e3->isNew = true;
          new_edges.push_back(e2);
          new_edges.push_back(e3);
        } else {
          VertexIter v = mesh.splitEdge(e);
          v->position = e->newPosition;
          v->isNew = true;
          HalfedgeIter h = v->halfedge();
          EdgeIter e0 = h->edge();
          EdgeIter e1 = h->next()->next()->edge();
          EdgeIter e2 = h->twin()->next()->edge();
          e0->isNew = true;
          e1->isNew = true;
          e2->isNew = true;
          new_edges.push_back(e1);


        }
			}
		}

		// dlip the old<->new edges 
		for (int i = 0; i < new_edges.size(); ++i) {
			HalfedgeCIter h = new_edges[i]->halfedge();
			VertexCIter v0 = h->vertex();
			VertexCIter v1 = h->twin()->vertex();
			if (v0->isNew != v1->isNew)
				mesh.flipEdge(new_edges[i]);
		}

		for (VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v) {
			if (!v->isNew) v->position = v->newPosition;
		}
   



    #endif


    

  }
}
